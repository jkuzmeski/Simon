#!/usr/bin/env python3
"""
Quick test to validate torque motion loader functionality
"""

import sys
import os
sys.path.append('Movement')

try:
    from torque_motion_loader import MotionLoader as TorqueMotionLoader
    print("✅ TorqueMotionLoader imported successfully")
    
    # Test loading the torque motion file
    motion_file = "logs/skrl/Biomech/2025-07-10_15-12-07_Test1/torque_profiles/torque_motion_20250710_164050.npz"
    
    if os.path.exists(motion_file):
        print(f"✅ Motion file exists: {motion_file}")
        
        # Test loading
        loader = TorqueMotionLoader(motion_file, "cpu")
        print(f"✅ Motion loaded successfully")
        print(f"   - Has torque data: {loader.has_torque_data}")
        print(f"   - Number of frames: {loader.num_frames}")
        print(f"   - Number of DOFs: {loader.num_dofs}")
        print(f"   - Duration: {loader.duration:.2f} seconds")
        
        if loader.has_torque_data:
            print(f"   - Torque shape: {loader.joint_torques.shape}")
            
        # Test sampling with torques
        if loader.has_torque_data:
            result = loader.sample_with_torques(num_samples=4)
            print(f"✅ Sample with torques successful - {len(result)} outputs")
            print(f"   - Torques shape: {result[-1].shape}")  # Last output is torques
        else:
            print("ℹ️  No torque data available for sampling")
        
    else:
        print(f"❌ Motion file not found: {motion_file}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
