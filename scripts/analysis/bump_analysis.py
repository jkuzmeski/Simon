#!/usr/bin/env python3
"""
Bump Event Analysis Script

This script analyzes the correlation between random force bumps applied to the pelvis
and the resulting changes in IMU data and biomechanical responses.

Usage:
    python bump_analysis.py --biomechanics_csv path/to/biomechanics_data.csv --bump_csv path/to/bump_events.csv
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def load_data(biomechanics_path: str, bump_path: str):
    """Load biomechanics and bump event data."""
    try:
        bio_df = pd.read_csv(biomechanics_path)
        bump_df = pd.read_csv(bump_path)
        print(f"Loaded {len(bio_df)} biomechanics data points")
        print(f"Loaded {len(bump_df)} bump events")
        return bio_df, bump_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def analyze_imu_response(bio_df: pd.DataFrame, bump_df: pd.DataFrame, window_size: int = 300):
    """Analyze IMU response around bump events."""
    
    results = []
    
    for _, bump in bump_df.iterrows():
        bump_timestep = bump['timestep']
        
        # Define window around bump event
        start_time = bump_timestep - window_size // 2
        end_time = bump_timestep + window_size // 2
        
        # Get biomechanics data in this window
        window_data = bio_df[(bio_df['timestep'] >= start_time) & 
                            (bio_df['timestep'] <= end_time)].copy()
        
        if len(window_data) < 10:  # Skip if not enough data
            continue
            
        # Add relative time from bump
        window_data['time_from_bump'] = window_data['timestep'] - bump_timestep
        
        # Extract IMU data (assuming standard naming)
        imu_cols = [col for col in window_data.columns if 'imu' in col.lower()]
        
        # Calculate metrics
        pre_bump = window_data[window_data['time_from_bump'] < 0]
        post_bump = window_data[window_data['time_from_bump'] >= 0]
        
        if len(pre_bump) > 0 and len(post_bump) > 0:
            analysis = {
                'bump_timestep': bump_timestep,
                'bump_force_magnitude': bump['force_magnitude'],
                'bump_force_x': bump['force_x'],
                'bump_force_y': bump['force_y'],
                'bump_force_z': bump['force_z'],
            }
            
            # Analyze each IMU column
            for col in imu_cols:
                if col in window_data.columns:
                    pre_mean = pre_bump[col].mean()
                    post_mean = post_bump[col].mean()
                    pre_std = pre_bump[col].std()
                    post_std = post_bump[col].std()
                    
                    analysis[f'{col}_pre_mean'] = pre_mean
                    analysis[f'{col}_post_mean'] = post_mean
                    analysis[f'{col}_change'] = post_mean - pre_mean
                    analysis[f'{col}_pre_std'] = pre_std
                    analysis[f'{col}_post_std'] = post_std
                    
            # Add pelvis position data if available
            if 'pelvis_x' in bump:
                analysis['pelvis_x'] = bump['pelvis_x']
                analysis['pelvis_y'] = bump['pelvis_y'] 
                analysis['pelvis_z'] = bump['pelvis_z']
                
            results.append(analysis)
    
    return pd.DataFrame(results)


def plot_bump_timeline(bump_df: pd.DataFrame, bio_df: pd.DataFrame):
    """Plot timeline showing bump events and key metrics."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Bump events over time
    axes[0].scatter(bump_df['timestep'], bump_df['force_magnitude'], 
                   c='red', s=60, alpha=0.7, label='Bump Events')
    axes[0].set_ylabel('Force Magnitude (N)')
    axes[0].set_title('Bump Events Timeline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: IMU linear acceleration (if available)
    imu_lin_cols = [col for col in bio_df.columns if 'imu' in col.lower() and 'lin_acc' in col.lower()]
    if imu_lin_cols:
        for col in imu_lin_cols[:3]:  # Plot first 3 components
            axes[1].plot(bio_df['timestep'], bio_df[col], alpha=0.7, label=col)
        
        # Mark bump events
        for _, bump in bump_df.iterrows():
            axes[1].axvline(x=bump['timestep'], color='red', alpha=0.5, linestyle='--')
            
        axes[1].set_ylabel('Linear Acceleration')
        axes[1].set_title('IMU Linear Acceleration with Bump Events')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No IMU linear acceleration data found', 
                    transform=axes[1].transAxes, ha='center', va='center')
    
    # Plot 3: IMU angular velocity (if available)
    imu_ang_cols = [col for col in bio_df.columns if 'imu' in col.lower() and 'ang_vel' in col.lower()]
    if imu_ang_cols:
        for col in imu_ang_cols[:3]:  # Plot first 3 components
            axes[2].plot(bio_df['timestep'], bio_df[col], alpha=0.7, label=col)
        
        # Mark bump events
        for _, bump in bump_df.iterrows():
            axes[2].axvline(x=bump['timestep'], color='red', alpha=0.5, linestyle='--')
            
        axes[2].set_ylabel('Angular Velocity')
        axes[2].set_title('IMU Angular Velocity with Bump Events')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No IMU angular velocity data found', 
                    transform=axes[2].transAxes, ha='center', va='center')
    
    axes[2].set_xlabel('Timestep')
    plt.tight_layout()
    return fig


def plot_bump_response_analysis(analysis_df: pd.DataFrame):
    """Plot analysis of IMU responses to bump events."""
    
    if len(analysis_df) == 0:
        print("No bump response data to plot")
        return None
        
    # Find IMU change columns
    change_cols = [col for col in analysis_df.columns if '_change' in col and 'imu' in col.lower()]
    
    if not change_cols:
        print("No IMU change data found")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: Force magnitude vs IMU response
    for i, col in enumerate(change_cols[:4]):  # Plot up to 4 IMU channels
        if i < len(axes):
            axes[i].scatter(analysis_df['bump_force_magnitude'], analysis_df[col], alpha=0.7)
            axes[i].set_xlabel('Bump Force Magnitude (N)')
            axes[i].set_ylabel(f'{col}')
            axes[i].set_title(f'Force vs {col}')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = analysis_df['bump_force_magnitude'].corr(analysis_df[col])
            axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_bump_direction_analysis(bio_df: pd.DataFrame, bump_df: pd.DataFrame, window_size: int = 150):
    """Analyze how bump direction affects response."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create bump direction categories
    bump_df = bump_df.copy()
    bump_df['force_direction'] = np.arctan2(bump_df['force_y'], bump_df['force_x']) * 180 / np.pi
    bump_df['direction_category'] = pd.cut(bump_df['force_direction'], 
                                          bins=[-180, -90, 0, 90, 180],
                                          labels=['Back', 'Left', 'Front', 'Right'])
    
    # Find a representative IMU column for analysis
    imu_cols = [col for col in bio_df.columns if 'imu' in col.lower()]
    if not imu_cols:
        print("No IMU data found for direction analysis")
        return None
        
    target_col = imu_cols[0]  # Use first IMU column
    
    # Analyze response by direction
    responses_by_direction = {}
    
    for direction in bump_df['direction_category'].unique():
        if pd.isna(direction):
            continue
            
        direction_bumps = bump_df[bump_df['direction_category'] == direction]
        direction_responses = []
        
        for _, bump in direction_bumps.iterrows():
            bump_timestep = bump['timestep']
            start_time = bump_timestep - window_size // 2
            end_time = bump_timestep + window_size // 2
            
            window_data = bio_df[(bio_df['timestep'] >= start_time) & 
                               (bio_df['timestep'] <= end_time)]
            
            if len(window_data) > 0 and target_col in window_data.columns:
                window_data = window_data.copy()
                window_data['time_from_bump'] = window_data['timestep'] - bump_timestep
                direction_responses.append(window_data)
        
        if direction_responses:
            responses_by_direction[direction] = pd.concat(direction_responses, ignore_index=True)
    
    # Plot responses by direction
    colors = ['blue', 'green', 'red', 'orange']
    for i, (direction, data) in enumerate(responses_by_direction.items()):
        if len(data) > 0:
            # Average response
            avg_response = data.groupby('time_from_bump')[target_col].mean()
            axes[0, 0].plot(avg_response.index, avg_response.values, 
                           color=colors[i % len(colors)], label=f'{direction}', linewidth=2)
    
    axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Bump Event')
    axes[0, 0].set_xlabel('Time from Bump (timesteps)')
    axes[0, 0].set_ylabel(f'{target_col}')
    axes[0, 0].set_title(f'Average {target_col} Response by Bump Direction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Force direction distribution
    axes[0, 1].hist(bump_df['force_direction'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Force Direction (degrees)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Bump Force Directions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Force magnitude by direction
    valid_data = bump_df.dropna(subset=['direction_category'])
    if len(valid_data) > 0:
        sns.boxplot(data=valid_data, x='direction_category', y='force_magnitude', ax=axes[1, 0])
        axes[1, 0].set_title('Force Magnitude by Direction')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recovery time analysis (simplified)
    axes[1, 1].text(0.5, 0.5, 'Recovery Time Analysis\n(Requires longer simulation)', 
                   transform=axes[1, 1].transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


def generate_summary_report(bio_df: pd.DataFrame, bump_df: pd.DataFrame, analysis_df: pd.DataFrame):
    """Generate a summary report of the bump analysis."""
    
    print("\n" + "="*60)
    print("BUMP EVENT ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    print(f"\nData Overview:")
    print(f"  Total biomechanics data points: {len(bio_df):,}")
    print(f"  Total bump events: {len(bump_df):,}")
    print(f"  Simulation duration: {bio_df['timestep'].max() - bio_df['timestep'].min():,} timesteps")
    
    if len(bump_df) > 0:
        print(f"\nBump Event Statistics:")
        print(f"  Average force magnitude: {bump_df['force_magnitude'].mean():.1f} ± {bump_df['force_magnitude'].std():.1f} N")
        print(f"  Force range: {bump_df['force_magnitude'].min():.1f} - {bump_df['force_magnitude'].max():.1f} N")
        
        # Calculate average intervals
        if len(bump_df) > 1:
            intervals = np.diff(bump_df['timestep'].values)
            avg_interval = intervals.mean()
            print(f"  Average interval between bumps: {avg_interval:.0f} timesteps ({avg_interval/60:.1f} seconds @ 60Hz)")
    
    # IMU data availability
    imu_cols = [col for col in bio_df.columns if 'imu' in col.lower()]
    print(f"\nIMU Data Availability:")
    print(f"  Total IMU channels found: {len(imu_cols)}")
    if imu_cols:
        print(f"  IMU channels: {', '.join(imu_cols[:5])}{'...' if len(imu_cols) > 5 else ''}")
    
    if len(analysis_df) > 0:
        change_cols = [col for col in analysis_df.columns if '_change' in col]
        print(f"\nIMU Response Analysis:")
        print(f"  Bump events with sufficient data: {len(analysis_df)}")
        print(f"  IMU change metrics calculated: {len(change_cols)}")
        
        # Find strongest correlations
        if len(change_cols) > 0:
            print(f"\nStrongest Force-Response Correlations:")
            for col in change_cols[:3]:  # Top 3
                if col in analysis_df.columns:
                    corr = analysis_df['bump_force_magnitude'].corr(analysis_df[col])
                    print(f"  {col}: r = {corr:.3f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze bump events and IMU response data')
    parser.add_argument('--biomechanics_csv', required=True, help='Path to biomechanics CSV file')
    parser.add_argument('--bump_csv', required=True, help='Path to bump events CSV file')
    parser.add_argument('--output_dir', default='bump_analysis_output', help='Output directory for plots')
    parser.add_argument('--window_size', type=int, default=300, help='Analysis window size around bumps (timesteps)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Load data
    bio_df, bump_df = load_data(args.biomechanics_csv, args.bump_csv)
    if bio_df is None or bump_df is None:
        return
    
    # Perform analysis
    print("\nAnalyzing IMU response to bump events...")
    analysis_df = analyze_imu_response(bio_df, bump_df, args.window_size)
    
    # Generate plots
    print("Generating plots...")
    
    # Timeline plot
    fig1 = plot_bump_timeline(bump_df, bio_df)
    if fig1:
        fig1.savefig(output_dir / 'bump_timeline.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'bump_timeline.png'}")
    
    # Response analysis plot
    fig2 = plot_bump_response_analysis(analysis_df)
    if fig2:
        fig2.savefig(output_dir / 'bump_response_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'bump_response_analysis.png'}")
    
    # Direction analysis plot
    fig3 = plot_bump_direction_analysis(bio_df, bump_df, args.window_size)
    if fig3:
        fig3.savefig(output_dir / 'bump_direction_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'bump_direction_analysis.png'}")
    
    # Save analysis results
    if len(analysis_df) > 0:
        analysis_path = output_dir / 'bump_analysis_results.csv'
        analysis_df.to_csv(analysis_path, index=False)
        print(f"  Saved: {analysis_path}")
    
    # Generate summary report
    generate_summary_report(bio_df, bump_df, analysis_df)
    
    print(f"\nAnalysis complete! Check the output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()