#!/usr/bin/env python3

import depthai as dai
import numpy as np
from PIL import Image
import time

print("Taking snapshot with OAK-D camera...")

try:
    # Create pipeline
    pipeline = dai.Pipeline()
    
    # Create color camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    
    xoutRgb.setStreamName("rgb")
    
    # Properties
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Linking
    camRgb.preview.link(xoutRgb.input)
    
    # Connect to device
    with dai.Device(pipeline) as device:
        print("Connected!")
        
        # Output queue
        q = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        
        # Get a frame
        print("Capturing...")
        for i in range(10):  # Try a few times
            inRgb = q.get()
            if inRgb is not None:
                # Get the frame as numpy array
                frame = inRgb.getCvFrame()
                
                # Convert BGR to RGB (if needed) and rotate
                rgb_frame = frame[:, :, ::-1]  # BGR to RGB
                
                # Rotate 180 degrees since camera is upside down
                rotated = np.rot90(rgb_frame, 2)
                
                # Save using PIL (no OpenCV needed)
                img = Image.fromarray(rotated.astype('uint8'), 'RGB')
                filename = f"oak_snapshot_{int(time.time())}.jpg"
                img.save(filename)
                
                print(f"✓ Saved {filename}")
                print(f"  Shape: {rotated.shape}")
                print(f"  Size: {img.size}")
                break
            
            time.sleep(0.1)
        else:
            print("Failed to capture frame")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
