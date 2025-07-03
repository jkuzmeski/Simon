# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Biomechanical actuator models for human-like control."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.actuators.actuator_pd import IdealPDActuator
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg


@dataclass
class BiomechActuatorCfg:
    """Configuration for biomechanical actuator model."""
    
    # Base actuator config
    base_cfg: IdealPDActuatorCfg
    
    # Biomechanical parameters
    enable_muscle_fatigue: bool = False
    fatigue_time_constant: float = 30.0  # seconds
    
    enable_velocity_scaling: bool = True
    velocity_scale_factor: float = 10.0
    
    enable_length_tension: bool = False
    optimal_joint_angles: dict[str, float] | None = None
    
    enable_activation_dynamics: bool = True
    activation_time_constant: float = 0.05  # seconds


class BiomechActuator(IdealPDActuator):
    """Biomechanical actuator model with muscle-like properties.
    
    This actuator extends the IdealPDActuator with biological muscle properties:
    - Velocity-dependent force scaling (Hill model)
    - Length-tension relationship
    - Activation dynamics
    - Muscle fatigue
    """
    
    cfg: BiomechActuatorCfg
    """The configuration for the biomechanical actuator model."""
    
    def __init__(self, cfg: BiomechActuatorCfg, *args, **kwargs):
        # Initialize base actuator
        super().__init__(cfg.base_cfg, *args, **kwargs)
        self.biomech_cfg = cfg
        
        # Initialize biomechanical state variables
        self._muscle_activation = torch.ones_like(self.computed_effort)
        self._fatigue_level = torch.zeros_like(self.computed_effort)
        self._prev_activation_command = torch.zeros_like(self.computed_effort)
        
        # Time step for dynamics
        self._dt = 1.0 / 60.0  # Assume 60 Hz, should be set from simulation
        
    def reset(self, env_ids: Sequence[int]):
        """Reset biomechanical state variables."""
        super().reset(env_ids)
        
        if env_ids is None:
            self._muscle_activation.fill_(1.0)
            self._fatigue_level.fill_(0.0)
            self._prev_activation_command.fill_(0.0)
        else:
            self._muscle_activation[env_ids] = 1.0
            self._fatigue_level[env_ids] = 0.0
            self._prev_activation_command[env_ids] = 0.0
    
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute biomechanical actuator forces."""
        
        # First compute base PD torques
        control_action = super().compute(control_action, joint_pos, joint_vel)
        
        # Apply biomechanical modulations
        bio_torques = self._apply_biomechanical_effects(
            control_action.joint_efforts, joint_pos, joint_vel
        )
        
        # Update internal states
        self._update_biomech_states(control_action.joint_efforts, joint_vel)
        
        # Set modified torques
        control_action.joint_efforts = bio_torques
        self.applied_effort = bio_torques
        
        return control_action
    
    def _apply_biomechanical_effects(
        self, torques: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """Apply various biomechanical effects to modify torques."""
        
        modified_torques = torques.clone()
        
        # 1. Velocity-dependent scaling (Hill muscle model)
        if self.biomech_cfg.enable_velocity_scaling:
            velocity_factor = self._compute_velocity_scaling(joint_vel)
            modified_torques = modified_torques * velocity_factor
        
        # 2. Length-tension relationship
        if self.biomech_cfg.enable_length_tension:
            length_factor = self._compute_length_tension(joint_pos)
            modified_torques = modified_torques * length_factor
        
        # 3. Activation dynamics
        if self.biomech_cfg.enable_activation_dynamics:
            modified_torques = modified_torques * self._muscle_activation
        
        # 4. Muscle fatigue
        if self.biomech_cfg.enable_muscle_fatigue:
            fatigue_factor = 1.0 - self._fatigue_level
            modified_torques = modified_torques * fatigue_factor
        
        return modified_torques
    
    def _compute_velocity_scaling(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Compute velocity-dependent force scaling (Hill model)."""
        # Normalized velocity (higher values = lower force capacity)
        v_norm = torch.abs(joint_vel) / self.biomech_cfg.velocity_scale_factor
        
        # Hill equation approximation: F = F_max * (1 - v/v_max) for concentric
        # For eccentric (lengthening), muscles can produce more force
        concentric_mask = torch.sign(joint_vel) == torch.sign(self.computed_effort)
        
        # Concentric scaling (force decreases with velocity)
        concentric_scale = torch.clamp(1.0 - v_norm, min=0.1, max=1.0)
        
        # Eccentric scaling (force can exceed isometric)
        eccentric_scale = torch.clamp(1.0 + 0.3 * v_norm, min=1.0, max=1.5)
        
        return torch.where(concentric_mask, concentric_scale, eccentric_scale)
    
    def _compute_length_tension(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Compute length-tension relationship."""
        if self.biomech_cfg.optimal_joint_angles is None:
            return torch.ones_like(joint_pos)
        
        # Simple gaussian around optimal angle
        optimal_angles = torch.zeros_like(joint_pos)
        for i, joint_name in enumerate(self._joint_names):
            if joint_name in self.biomech_cfg.optimal_joint_angles:
                optimal_angles[:, i] = self.biomech_cfg.optimal_joint_angles[joint_name]
        
        # Gaussian scaling around optimal length
        angle_dev = torch.abs(joint_pos - optimal_angles)
        length_factor = torch.exp(-0.5 * (angle_dev / 0.5) ** 2)  # sigma = 0.5 rad
        
        return torch.clamp(length_factor, min=0.3, max=1.0)
    
    def _update_biomech_states(self, commanded_torques: torch.Tensor, joint_vel: torch.Tensor):
        """Update internal biomechanical state variables."""
        
        # Update activation dynamics
        if self.biomech_cfg.enable_activation_dynamics:
            tau_act = self.biomech_cfg.activation_time_constant
            # Target activation based on commanded torque magnitude
            target_activation = torch.tanh(torch.abs(commanded_torques) / self.effort_limit)
            
            # First-order dynamics
            activation_rate = (target_activation - self._muscle_activation) / tau_act
            self._muscle_activation += activation_rate * self._dt
            self._muscle_activation = torch.clamp(self._muscle_activation, min=0.0, max=1.0)
        
        # Update fatigue
        if self.biomech_cfg.enable_muscle_fatigue:
            tau_fatigue = self.biomech_cfg.fatigue_time_constant
            # Fatigue accumulates with high activation
            fatigue_rate = self._muscle_activation / tau_fatigue
            # Recovery when not active
            recovery_rate = (1.0 - self._muscle_activation) / (tau_fatigue * 2.0)
            
            self._fatigue_level += (fatigue_rate - recovery_rate) * self._dt
            self._fatigue_level = torch.clamp(self._fatigue_level, min=0.0, max=0.8)  # Max 80% fatigue
    
    def set_simulation_dt(self, dt: float):
        """Set the simulation time step for dynamics integration."""
        self._dt = dt
