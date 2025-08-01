#!/usr/bin/env python3
"""
URDF Fixed Joint to Revolute Converter

This script converts specified fixed joints to revolute joints with very tight limits
to prevent Isaac Sim from merging the connected links while maintaining essentially
the same physical behavior.

This is only needed if you specifically need separate collision bodies for
certain links (e.g., foot contacts, sensor attachments).
"""

import xml.etree.ElementTree as ET
import argparse
from typing import List
import sys

class URDFJointConverter:
    def __init__(self, tight_limit: float = 0.001, effort_limit: float = 1000.0):
        """
        Initialize the joint converter.
        
        Args:
            tight_limit: Very small joint limit in radians (default: 0.001 rad ≈ 0.06°)
            effort_limit: High effort limit to prevent movement (default: 1000 N⋅m)
        """
        self.tight_limit = tight_limit
        self.effort_limit = effort_limit
        
    def parse_urdf(self, urdf_path: str) -> ET.ElementTree:
        """Parse the URDF file and return the XML tree."""
        try:
            tree = ET.parse(urdf_path)
            return tree
        except ET.ParseError as e:
            print(f"Error parsing URDF file: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"URDF file not found: {urdf_path}")
            sys.exit(1)
    
    def find_joints_by_child_links(self, tree: ET.ElementTree, 
                                  child_link_names: List[str]) -> List[ET.Element]:
        """Find all joints that have the specified links as children."""
        root = tree.getroot()
        target_joints = []
        
        for joint in root.findall('joint'):
            child = joint.find('child')
            if child is not None:
                child_name = child.get('link')
                if child_name in child_link_names:
                    target_joints.append(joint)
        
        return target_joints
    
    def convert_fixed_to_revolute(self, joint: ET.Element) -> bool:
        """
        Convert a fixed joint to a revolute joint with tight limits.
        
        Returns:
            True if conversion was performed, False if joint was not fixed
        """
        joint_name = joint.get('name', 'unknown')
        joint_type = joint.get('type', 'unknown')
        
        if joint_type != 'fixed':
            print(f"  Skipping {joint_name} (type: {joint_type})")
            return False
        
        print(f"  Converting {joint_name} from fixed to revolute")
        
        # Change joint type
        joint.set('type', 'revolute')
        
        # Add axis if not present (default to z-axis)
        axis = joint.find('axis')
        if axis is None:
            axis = ET.SubElement(joint, 'axis')
            axis.set('xyz', '0 0 1')
        
        # Add very tight limits
        limit = joint.find('limit')
        if limit is None:
            limit = ET.SubElement(joint, 'limit')
        
        limit.set('lower', f"{-self.tight_limit:.6f}")
        limit.set('upper', f"{self.tight_limit:.6f}")
        limit.set('effort', f"{self.effort_limit}")
        limit.set('velocity', "0.1")  # Low velocity limit
        
        return True
    
    def process_urdf(self, input_path: str, output_path: str, 
                     links_to_preserve: List[str] = None) -> None:
        """
        Main processing function to convert fixed joints to revolute.
        
        Args:
            input_path: Path to input URDF file
            output_path: Path to output URDF file
            links_to_preserve: List of child link names whose joints should be converted.
                              If None, uses common problematic links for Isaac Sim.
        """
        if links_to_preserve is None:
            # Default problematic links that Isaac Sim typically merges
            links_to_preserve = [
                'lffoot', 'rffoot', 'lbfoot', 'rbfoot',  # Foot links
                'lpf1', 'lpf2', 'rpf1', 'rpf2', 'lpb1', 'lpb2', 'rpb1', 'rpb2',  # Plate links
                'lidar_link', 'imu_link',  # Sensor links
                # 'base_inertia', 'yellow_plate'  # You might want to keep these merged
            ]
        
        print(f"Processing URDF: {input_path}")
        print(f"Target minimum joint limit: ±{self.tight_limit:.6f} rad (±{self.tight_limit*57.3:.2f}°)")
        print(f"Links to preserve as separate bodies: {len(links_to_preserve)}")
        
        # Parse URDF
        tree = self.parse_urdf(input_path)
        
        # Find target joints
        target_joints = self.find_joints_by_child_links(tree, links_to_preserve)
        
        if not target_joints:
            print("\nNo target joints found!")
            return
        
        print(f"\nFound {len(target_joints)} joints to potentially convert:")
        
        # Convert joints
        converted_count = 0
        for joint in target_joints:
            if self.convert_fixed_to_revolute(joint):
                converted_count += 1
        
        # Write output
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"\n✅ Processing complete!")
        print(f"Converted {converted_count} fixed joints to revolute")
        print(f"Modified URDF written to: {output_path}")
        
        if converted_count > 0:
            print(f"\n🎯 These links should now remain separate in Isaac Sim!")
            print(f"Note: The joints have very tight limits (±{self.tight_limit*57.3:.2f}°) so behavior should be nearly identical.")
        else:
            print(f"\n⚠️  No fixed joints were converted. Check if the target links exist.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fixed joints to revolute joints with tight limits to prevent Isaac Sim link merging"
    )
    parser.add_argument('input_urdf', help='Input URDF file path')
    parser.add_argument('output_urdf', help='Output URDF file path')
    parser.add_argument('--limit', type=float, default=0.001,
                       help='Joint limit in radians (default: 0.001 rad ≈ 0.06°)')
    parser.add_argument('--effort', type=float, default=1000.0,
                       help='Effort limit in N⋅m (default: 1000)')
    parser.add_argument('--links', nargs='+', 
                       help='Specific child link names to convert (space-separated)')
    parser.add_argument('--feet-only', action='store_true',
                       help='Only convert foot links (lffoot, rffoot, lbfoot, rbfoot)')
    parser.add_argument('--sensors-only', action='store_true',
                       help='Only convert sensor links (lidar_link, imu_link)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.limit <= 0:
        print("Error: Joint limit must be positive")
        sys.exit(1)
    
    if args.effort <= 0:
        print("Error: Effort limit must be positive")
        sys.exit(1)
    
    # Determine target links
    target_links = None
    if args.links:
        target_links = args.links
    elif args.feet_only:
        target_links = ['lffoot', 'rffoot', 'lbfoot', 'rbfoot']
    elif args.sensors_only:
        target_links = ['lidar_link', 'imu_link']
    # Otherwise use default (None = all common problematic links)
    
    converter = URDFJointConverter(tight_limit=args.limit, effort_limit=args.effort)
    converter.process_urdf(args.input_urdf, args.output_urdf, target_links)


if __name__ == "__main__":
    main()