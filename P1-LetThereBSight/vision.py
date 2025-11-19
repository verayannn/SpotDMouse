import depthai as dai
import cv2
import time

print("Testing OAK-D camera...")

try:
    # Create pipeline
    pipeline = dai.Pipeline()

    # Create camera node (new API)
    cam = pipeline.createCamera()
    xout = pipeline.createXLinkOut()
    
    xout.setStreamName("rgb")
    
    # Camera properties
    cam.setPreviewSize(640, 480)
    cam.setFps(30)
    
    # Linking
    cam.preview.link(xout.input)
    
    # Connect and capture
    with dai.Device(pipeline) as device:
        print("Connected to camera!")
        
        # Get RGB queue
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        # Wait for a frame
        print("Waiting for frame...")
        while True:
            data = q.get()
            if data is not None:
                frame = data.getCvFrame()
                
                # Rotate 180 degrees if camera is upside down
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Save snapshot
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                print(f"Image shape: {frame.shape}")
                break
                
except Exception as e:
    print(f"Error: {e}")
    
    # Try alternative API if above fails
    print("\nTrying alternative API...")
    try:
        # Alternative for newest API
        pipeline2 = dai.Pipeline()
        
        # Create nodes
        camRgb = pipeline2.create(dai.node.Camera)
        xoutRgb = pipeline2.create(dai.node.XLinkOut)
        
        xoutRgb.setStreamName("rgb")
        
        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.CameraProperties.SensorResolution.THE_1080_P)
        camRgb.setPreviewSize(640, 480)
        camRgb.setInterleaved(False)
        
        # Linking
        camRgb.preview.link(xoutRgb.input)
        
        with dai.Device(pipeline2) as device:
            print("Connected with alternative API!")
            q = device.getOutputQueue(name="rgb")
            
            data = q.get()
            frame = data.getCvFrame()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            filename = f"snapshot_alt_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nCheck your depthai version:")
        print(f"Current version: {dai.__version__}")