# Updated Gesture Controller with Enhanced Instructions and Detection
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import time
from collections import deque

class GestureController:
    """Enhanced Gesture Controller with improved detection and user feedback"""
    
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture recognition state
        self.gesture_buffer = deque(maxlen=15)  # Buffer for gesture smoothing
        self.last_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_threshold = 0.6
        
        # Control flags
        self.is_running = False
        self.detection_paused = False
        self.show_instructions = True
        
        # Gesture mappings with descriptions
        self.gesture_actions = {
            'right_fist': {
                'action': self._next_question,
                'description': '🤜 Right Fist: Next question',
                'icon': '⏭️'
            },
            'left_fist': {
                'action': self._repeat_question,
                'description': '🤛 Left Fist: Repeat question',
                'icon': '🔄'
            },
            'open_palm': {
                'action': self._pause_resume,
                'description': '✋ Open Palm: Pause/Resume',
                'icon': '⏸️▶️'
            },
            'two_fingers': {
                'action': self._show_rubric,
                'description': '✌️ Two Fingers: Show rubric',
                'icon': '📋'
            },
            'thumbs_up': {
                'action': self._submit_answer,
                'description': '👍 Thumbs Up: Submit answer',
                'icon': '✅'
            },
            'stop_hand': {
                'action': self._emergency_stop,
                'description': '🛑 Stop Hand: Emergency stop',
                'icon': '🚨'
            }
        }
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def get_gesture_instructions(self):
        """Return formatted gesture instructions"""
        instructions = []
        for gesture, info in self.gesture_actions.items():
            instructions.append(f"{info['icon']} {info['description']}")
        return instructions
    
    def start_detection(self):
        """Start gesture detection in a separate thread"""
        self.is_running = True
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()
    
    def stop_detection(self):
        """Stop gesture detection"""
        self.is_running = False
    
    def pause_detection(self):
        """Pause gesture detection temporarily"""
        self.detection_paused = True
    
    def resume_detection(self):
        """Resume gesture detection"""
        self.detection_paused = False
    
    def _detection_loop(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access camera")
            return
        
        print("Gesture detection started. Press 'q' to quit, 'p' to pause/resume.")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not self.detection_paused:
                # Process frame for hand detection
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Detect gesture
                        gesture = self._classify_gesture(hand_landmarks)
                        if gesture:
                            self.gesture_buffer.append(gesture)
                            
                            # Check for consistent gesture
                            if self._is_gesture_stable(gesture):
                                self._execute_gesture_action(gesture)
                else:
                    self.gesture_buffer.clear()
            
            # Add status overlay
            self._draw_status_overlay(frame)
            
            # Display frame
            cv2.imshow('Excel Interview - Gesture Control', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.detection_paused = not self.detection_paused
                print(f"Detection {'paused' if self.detection_paused else 'resumed'}")
            elif key == ord('i'):
                self.show_instructions = not self.show_instructions
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _classify_gesture(self, landmarks):
        """Classify hand gesture from landmarks"""
        # Extract landmark coordinates
        landmarks_array = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        # Calculate finger states (extended or folded)
        finger_states = self._get_finger_states(landmarks_array)
        
        # Classify based on finger states and hand position
        extended_fingers = sum(finger_states)
        
        if extended_fingers == 0:
            # Closed fist - determine left or right based on hand position
            wrist_x = landmarks_array[0][0]  # Wrist x-coordinate
            if wrist_x < 0.4:  # Left side of frame
                return 'left_fist'
            elif wrist_x > 0.6:  # Right side of frame
                return 'right_fist'
        
        elif extended_fingers == 5:
            # Open palm
            return 'open_palm'
        
        elif extended_fingers == 2:
            # Two fingers (peace sign or similar)
            if finger_states[1] and finger_states[2]:  # Index and middle
                return 'two_fingers'
        
        elif extended_fingers == 1:
            # One finger - check if thumb (thumbs up)
            if finger_states[0]:  # Thumb extended
                thumb_y = landmarks_array[4][1]  # Thumb tip
                wrist_y = landmarks_array[0][1]  # Wrist
                if thumb_y < wrist_y:  # Thumb pointing up
                    return 'thumbs_up'
        
        elif extended_fingers == 4 and not finger_states[2]:
            # Four fingers extended except middle (stop gesture)
            return 'stop_hand'
        
        return None
    
    def _get_finger_states(self, landmarks):
        """Determine which fingers are extended"""
        # Finger tip and pip landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        finger_states = []
        
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # Thumb (different logic)
                # Thumb extended if tip is further from wrist than pip
                extended = landmarks[tip][0] > landmarks[pip][0]  # Use x-coordinate for thumb
            else:
                # Other fingers extended if tip is higher than pip
                extended = landmarks[tip][1] < landmarks[pip][1]
            
            finger_states.append(extended)
        
        return finger_states
    
    def _is_gesture_stable(self, gesture):
        """Check if gesture is stable and consistent"""
        if len(self.gesture_buffer) < 10:
            return False
        
        # Count occurrences of current gesture in buffer
        gesture_count = self.gesture_buffer.count(gesture)
        confidence = gesture_count / len(self.gesture_buffer)
        
        # Update confidence and check threshold
        self.gesture_confidence = confidence
        return confidence > self.gesture_threshold and gesture != self.last_gesture
    
    def _execute_gesture_action(self, gesture):
        """Execute action for recognized gesture"""
        if gesture in self.gesture_actions:
            print(f"Gesture detected: {gesture} ({self.gesture_actions[gesture]['description']})")
            
            try:
                self.gesture_actions[gesture]['action']()
                self.last_gesture = gesture
                
                # Clear buffer after action
                self.gesture_buffer.clear()
                
                # Brief pause to prevent rapid-fire actions
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error executing gesture action: {e}")
    
    def _draw_status_overlay(self, frame):
        """Draw status information on frame"""
        h, w, _ = frame.shape
        
        # Status text
        status = "PAUSED" if self.detection_paused else "ACTIVE"
        status_color = (0, 0, 255) if self.detection_paused else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Gesture confidence
        if self.gesture_confidence > 0:
            cv2.putText(frame, f"Confidence: {self.gesture_confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions toggle
        cv2.putText(frame, "Press 'i' to toggle instructions", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show gesture instructions if enabled
        if self.show_instructions:
            self._draw_instruction_overlay(frame)
    
    def _draw_instruction_overlay(self, frame):
        """Draw gesture instructions on frame"""
        h, w, _ = frame.shape
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 350, 10), (w - 10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Instructions title
        cv2.putText(frame, "Gesture Controls:", (w - 340, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw each instruction
        y_offset = 60
        for gesture, info in self.gesture_actions.items():
            text = info['description'].split(': ')[1]  # Remove emoji part for overlay
            cv2.putText(frame, f"• {text}", (w - 335, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 25
    
    # Gesture action methods
    def _next_question(self):
        """Move to next question"""
        pyautogui.press('tab')  # Navigate to next button
        pyautogui.press('enter')  # Click next
    
    def _repeat_question(self):
        """Repeat current question"""
        pyautogui.hotkey('ctrl', 'r')  # Refresh page
    
    def _pause_resume(self):
        """Pause or resume interview"""
        self.detection_paused = not self.detection_paused
        print(f"Interview {'paused' if self.detection_paused else 'resumed'} via gesture")
    
    def _show_rubric(self):
        """Show scoring rubric (placeholder)"""
        print("Showing rubric (gesture detected)")
        # Could trigger a modal or sidebar display
    
    def _submit_answer(self):
        """Submit current answer"""
        pyautogui.hotkey('ctrl', 'enter')  # Submit shortcut
    
    def _emergency_stop(self):
        """Emergency stop interview"""
        print("Emergency stop activated!")
        self.stop_detection()
        pyautogui.hotkey('alt', 'f4')  # Close application

# Utility functions for Streamlit integration
def create_gesture_guide_markdown():
    """Create markdown guide for gesture controls"""
    controller = GestureController()
    instructions = controller.get_gesture_instructions()
    
    guide = """
## 🤝 Gesture Control Guide

Hold gestures clearly in front of the camera for 2-3 seconds for recognition.

### Navigation Gestures
- 🤜 **Right Fist**: Move to next question
- 🤛 **Left Fist**: Repeat current question  
- ✋ **Open Palm**: Pause/Resume interview

### Control Gestures
- ✌️ **Two Fingers**: Show scoring rubric
- 👍 **Thumbs Up**: Submit current answer
- 🛑 **Stop Hand**: Emergency stop interview

### Tips for Best Results
- 💡 Keep your hand clearly visible to camera
- 🔆 Ensure good lighting on your hands
- 📏 Position hand 1-2 feet from camera
- ⏱️ Hold gesture steady for 2-3 seconds
- 🔄 Wait for confirmation before next gesture

### Camera Controls
- Press **'p'** to pause/resume detection
- Press **'i'** to toggle instruction overlay
- Press **'q'** to quit gesture control
"""
    return guide

def get_gesture_status_info():
    """Get gesture control status information"""
    return {
        'supported_gestures': [
            'Right Fist (Next)', 'Left Fist (Repeat)', 'Open Palm (Pause/Resume)',
            'Two Fingers (Rubric)', 'Thumbs Up (Submit)', 'Stop Hand (Emergency)'
        ],
        'requirements': [
            'Working webcam', 'Good lighting', 'Clear hand visibility',
            'Python packages: opencv-python, mediapipe, pyautogui'
        ],
        'tips': [
            'Hold gestures for 2-3 seconds',
            'Position hands 1-2 feet from camera',
            'Ensure good lighting for better detection',
            'Wait for gesture confirmation before proceeding'
        ]
    }