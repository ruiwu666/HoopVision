"""
HoopVision: Basketball Analytics System
CS366 Final Project
Author: Rui Wu
Date: 2025-12-05
This project implements a basketball analytics system that detects players, referees, the ball, and the hoop in the video feed. 
It tracks ball possession, detects shots, and classifies players into teams based on jersey colors.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque
import math
import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# Paths
HOOP_MODEL_PATH = 'models/best.pt'
PLAYER_MODEL_PATH = 'models/player_detector.pt'
BALL_MODEL_PATH = 'models/ball_detector_model.pt'
VIDEO_PATH = '1.mp4'

# Detection Parameters
CONF_THRESHOLD_PLAYERS = 0.6
CONF_THRESHOLD_BALL = 0.35
MAX_BALL_TRAVEL_DIST = 150    
POSSESSION_DIST_THRESH = 110

# Color Classification Parameters
COLOR_DISTANCE_THRESHOLD = 50 # If color dist > this, it's a ref/outlier
# Ref threshold: sort by saturation
REF_SATURATION_THRESHOLD = 45 


COLOR_TEAM_A = (255, 0, 0)    # Blue
COLOR_TEAM_B = (0, 0, 255)    # Red
COLOR_REF    = (128, 128, 128) # Grey
COLOR_BALL   = (0, 255, 0)    # Bright Green
COLOR_HOOP   = (0, 255, 255)  # Yellow

# ==========================================
# LOGIC CLASSES
# ==========================================

class KalmanBallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.prediction = np.zeros((2, 1), np.float32)
        self.missed_frames = 0

    def update(self, x, y):
        self.missed_frames = 0
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        self.prediction = self.kf.predict()
        return int(self.prediction[0]), int(self.prediction[1])

    def predict(self):
        self.missed_frames += 1
        self.prediction = self.kf.predict()
        return int(self.prediction[0]), int(self.prediction[1])

class TeamClassifier:
    """
    Use 3-Means clustering: Automatically split everyone into 3 classes.
    1. The class with the lowest saturation -> Referee.
    2. The remaining two classes -> Team A and Team B.
    """
    def __init__(self):
        self.kmeans = None
        self.mapping = {} # {label_id: 'Ref'/'TeamA'/'TeamB'}

    def get_features(self, img_crop):
        """Extract features: Average HSV color"""
        h, w, _ = img_crop.shape
        # Crop out head and legs, keep torso only
        center_crop = img_crop[int(h*0.2):int(h*0.6), int(w*0.25):int(w*0.75)]
        if center_crop.size == 0: return np.zeros(3)
        
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        return avg_hsv 

    def fit_teams(self, player_crops):
        # Need enough samples to perform 3-class clustering
        if not player_crops or len(player_crops) < 5: 
            print("Warning: Not enough samples to calibrate teams.")
            return
        
        # 1. Extract features for all samples
        data = [self.get_features(c) for c in player_crops]
        
        # 2. Force clustering into 3 classes (Ref, A, B)
        self.kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        self.kmeans.fit(data)
        centers = self.kmeans.cluster_centers_ 

        # 3. Analyze cluster centers to decide who is Ref/TeamA/TeamB
        clusters_info = []
        for i, center in enumerate(centers):
            saturation = center[1]
            hue = center[0]
            clusters_info.append({'id': i, 'sat': saturation, 'hue': hue})
        
        # Sort by saturation
        clusters_info.sort(key=lambda x: x['sat'])
        
        ref_cluster = clusters_info[0] # Lowest saturation
        self.mapping[ref_cluster['id']] = 'Ref'
        
        # The remaining two are teams
        team_clusters = clusters_info[1:]
        
        # Sort by Hue to assign A/B
        team_clusters.sort(key=lambda x: x['hue'])
        
        self.mapping[team_clusters[0]['id']] = 'TeamB'  # Low Hue is Team B
        self.mapping[team_clusters[1]['id']] = 'TeamA'  # High Hue is Team A
        
        print(f"Classifier Mapped (Swapped A/B): {self.mapping}")
        print(f"Ref Saturation: {ref_cluster['sat']:.2f}")

    def predict(self, img_crop):
        if self.kmeans is None: return "TeamA"
        feat = self.get_features(img_crop)
        label = self.kmeans.predict([feat])[0]
        return self.mapping.get(label, "Unknown")

# ==========================================
# MAIN SYSTEM
# ==========================================

class BasketballAnalytics:
    def __init__(self):
        print("Loading models...")
        self.model_hoop = YOLO(HOOP_MODEL_PATH)
        self.model_player = YOLO(PLAYER_MODEL_PATH)
        self.model_ball = YOLO(BALL_MODEL_PATH)

        self.ball_tracker = KalmanBallTracker()
        self.team_classifier = TeamClassifier()
        self.teams_initialized = False
        
        self.players = {} 
        self.ball_history = deque(maxlen=20) 
        self.hoop_bbox = None
        
        self.stats = {
            "TeamA": { "FGM": 0, "FGA": 0, "PossessionFrames": 0 },
            "TeamB": { "FGM": 0, "FGA": 0, "PossessionFrames": 0 }
        }
        
        self.current_possession = None
        self.shot_active = False
        self.shooting_team = None
        self.shot_cooldown = 0
        self.frame_count = 0
        
        self.frame_height = 0
        self.frame_width = 0
        
        # possession stability
        self.pending_possession = None
        self.pending_frames = 0         # Consecutive frames detected
        self.POSSESSION_CONFIRM_FRAMES = 8  # Need 8 consecutive frames to switch

    def detect_objects(self, frame):
        r_hoop = self.model_hoop(frame, verbose=False, conf=0.5)[0]
        r_players = self.model_player(frame, verbose=False, conf=CONF_THRESHOLD_PLAYERS)[0]
        r_ball = self.model_ball(frame, verbose=False, conf=CONF_THRESHOLD_BALL)[0]
        return r_hoop, r_players, r_ball

    def filter_ball(self, ball_results, frame_shape):
        """Sanity check for ball detection"""
        valid_balls = []
        for box in ball_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2-x1) * (y2-y1)
            if area < 50 or area > 3500: continue # Size filter
            ratio = (x2-x1) / (y2-y1)
            if ratio < 0.6 or ratio > 1.6: continue
            center = ((x1+x2)//2, (y1+y2)//2)
            valid_balls.append(center)

        if not valid_balls: return None
        best_candidate = valid_balls[0]

        if len(self.ball_history) > 0:
            last_x, last_y = self.ball_history[-1]
            dist = math.hypot(best_candidate[0] - last_x, best_candidate[1] - last_y)
            if dist > MAX_BALL_TRAVEL_DIST: return None
        return best_candidate

    def is_valid_player_detection(self, x1, y1, x2, y2):
        """Filter out shot clock, scoreboard, and bad detections"""
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # 1. Filter Shot Clock
        if self.hoop_bbox:
            hy2 = self.hoop_bbox[3] # Hoop bottom
            if y2 < hy2: return False # Feet higher than hoop rim

        # 2. Filter Scoreboard
        if self.frame_height > 0 and center_y > self.frame_height * 0.88:
            return False
        
        # 3. Shape Filter
        if height > 0 and width > height * 1.5:
            return False
        
        # 4. Filter tiny boxes (hands/ball detected as person)
        if height < 80:
            return False
        if area < 4000:
            return False
            
        return True
    
    def filter_overlapping_detections(self, detections):
        """
        Filter out smaller boxes that are overlapping significantly with larger boxes
        """
        boxes = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            boxes.append({'box': (x1, y1, x2, y2), 'area': area, 'valid': True})
        
        for i in range(len(boxes)):
            if not boxes[i]['valid']: continue
            for j in range(len(boxes)):
                if i == j or not boxes[j]['valid']: continue
                
                # Calculate Overlap
                x1_i, y1_i, x2_i, y2_i = boxes[i]['box']
                x1_j, y1_j, x2_j, y2_j = boxes[j]['box']
                
                ix1 = max(x1_i, x1_j)
                iy1 = max(y1_i, y1_j)
                ix2 = min(x2_i, x2_j)
                iy2 = min(y2_i, y2_j)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    smaller_area = min(boxes[i]['area'], boxes[j]['area'])
                    
                    # If overlap > 50% of smaller box, remove smaller box
                    if inter_area > smaller_area * 0.5:
                        if boxes[i]['area'] < boxes[j]['area']:
                            boxes[i]['valid'] = False
                        else:
                            boxes[j]['valid'] = False
        
        return [b['box'] for b in boxes if b['valid']]

    def track_players(self, frame, detections):
        # First filter overlaps
        valid_boxes = self.filter_overlapping_detections(detections)
        
        for box_coords in valid_boxes:
            x1, y1, x2, y2 = box_coords
            center = ((x1+x2)//2, (y1+y2)//2)
            
            # Filter invalid objects
            if not self.is_valid_player_detection(x1, y1, x2, y2):
                continue
            
            # Exclude overlap with Hoop
            if self.hoop_bbox:
                hx1, hy1, hx2, hy2 = self.hoop_bbox
                ix1 = max(x1, hx1); iy1 = max(y1, hy1)
                ix2 = min(x2, hx2); iy2 = min(y2, hy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2-ix1)*(iy2-iy1)
                    hoop_area = (hx2-hx1)*(hy2-hy1)
                    if inter_area > hoop_area * 0.05: continue

            # Match Existing
            matched_id = None
            min_dist = 60.0
            for pid, pdata in self.players.items():
                if pdata['last_seen'] == self.frame_count: continue
                dist = math.hypot(center[0]-pdata['center'][0], center[1]-pdata['center'][1])
                if dist < min_dist:
                    min_dist = dist
                    matched_id = pid
            
            # Update/Create
            if matched_id is None:
                matched_id = len(self.players) + 1
                crop = frame[y1:y2, x1:x2]
                team = "TeamA"
                if self.teams_initialized and crop.size > 0:
                    team = self.team_classifier.predict(crop)
                
                self.players[matched_id] = {
                    'bbox': (x1,y1,x2,y2), 'center': center,
                    'team': team, 'last_seen': self.frame_count
                }
            else:
                self.players[matched_id]['bbox'] = (x1,y1,x2,y2)
                self.players[matched_id]['center'] = center
                self.players[matched_id]['last_seen'] = self.frame_count
                
                if self.frame_count % 15 == 0 and self.teams_initialized:
                     crop = frame[y1:y2, x1:x2]
                     if crop.size > 0:
                         self.players[matched_id]['team'] = self.team_classifier.predict(crop)

        # Cleanup old
        to_remove = [pid for pid, p in self.players.items() if self.frame_count - p['last_seen'] > 20]
        for pid in to_remove: del self.players[pid]

    def find_closest_player_to_ball(self, ball_center):
        """
        Determine possession:
        1. Find players who contain the ball in their bounding box (expanded)
        2. If none, check if ball is directly below (vertical alignment)
        3. Fallback to distance check
        """
        if ball_center is None: return None, float('inf')
        
        bx, by = ball_center
        
        # Step 1: Check overlapping boxes (with expansion)
        best_overlap_team = None
        best_overlap_dist = float('inf')
        
        for pid, p in self.players.items():
            if p['last_seen'] != self.frame_count: continue
            if p['team'] == "Ref": continue
            
            x1, y1, x2, y2 = p['bbox']
            # Expand box (ball might be dribbled low or to side)
            expand_x = 50
            expand_y = 60
            
            if (x1 - expand_x) < bx < (x2 + expand_x) and (y1) < by < (y2 + expand_y):
                dist = math.hypot(p['center'][0] - bx, p['center'][1] - by)
                if dist < best_overlap_dist:
                    best_overlap_dist = dist
                    best_overlap_team = p['team']
        
        if best_overlap_team:
            return best_overlap_team, best_overlap_dist
        
        # Step 2: Vertical alignment check
        vertical_matches = []
        for pid, p in self.players.items():
            if p['last_seen'] != self.frame_count: continue
            if p['team'] == "Ref": continue
            
            x1, y1, x2, y2 = p['bbox']
            player_width = x2 - x1
            
            margin = player_width * 0.3
            if (x1 - margin) < bx < (x2 + margin):
                if by > y2 - 50:
                    dist = math.hypot(p['center'][0] - bx, p['center'][1] - by)
                    vertical_matches.append((p['team'], dist))
        
        if vertical_matches:
            vertical_matches.sort(key=lambda x: x[1])
            return vertical_matches[0]
        
        # Step 3: Simple distance fallback
        min_dist = float('inf')
        closest_team = None
        for pid, p in self.players.items():
            if p['last_seen'] != self.frame_count: continue
            if p['team'] == "Ref": continue
            dist = math.hypot(p['center'][0] - bx, p['center'][1] - by)
            if dist < min_dist:
                min_dist = dist
                closest_team = p['team']
        
        return closest_team, min_dist

    def update_logic(self, ball_center):
        if ball_center is None: return

        # 1. POSSESSION (With Stability)
        closest_team, min_dist = self.find_closest_player_to_ball(ball_center)
        
        if min_dist < POSSESSION_DIST_THRESH and closest_team:
            if closest_team == self.current_possession:
                self.stats[closest_team]["PossessionFrames"] += 1
                self.pending_possession = None
                self.pending_frames = 0
            else:
                if closest_team == self.pending_possession:
                    self.pending_frames += 1
                    if self.pending_frames >= self.POSSESSION_CONFIRM_FRAMES:
                        self.current_possession = closest_team
                        self.stats[closest_team]["PossessionFrames"] += 1
                        self.pending_possession = None
                        self.pending_frames = 0
                else:
                    self.pending_possession = closest_team
                    self.pending_frames = 1
                
                if self.current_possession and self.pending_frames < self.POSSESSION_CONFIRM_FRAMES:
                    self.stats[self.current_possession]["PossessionFrames"] += 1

        # 2. SHOT DETECTION
        if self.shot_cooldown > 0: self.shot_cooldown -= 1
        
        if self.hoop_bbox and len(self.ball_history) > 3:
            hx1, hy1, hx2, hy2 = self.hoop_bbox
            bx, by = ball_center
            in_hoop_x = hx1 < bx < hx2
            dy = self.ball_history[-1][1] - self.ball_history[-3][1]

            # Attempt
            if not self.shot_active and self.shot_cooldown == 0 and in_hoop_x and dy < -3 and by < hy2 + 50:
                self.shot_active = True
                
                # Determine shooter
                shooter, _ = self.find_closest_player_to_ball(ball_center)
                if shooter:
                    self.shooting_team = shooter
                else:
                    self.shooting_team = self.current_possession
                
                if self.shooting_team and self.shooting_team in self.stats:
                    self.stats[self.shooting_team]["FGA"] += 1
                print(f"Shot Attempt by {self.shooting_team}!")

            # Made
            if self.shot_active:
                if dy > 3 and hx1 < bx < hx2 and hy1 < by < hy2:
                    print(f"SHOT MADE by {self.shooting_team}!")
                    if self.shooting_team and self.shooting_team in self.stats:
                        self.stats[self.shooting_team]["FGM"] += 1
                    self.shot_active = False
                    self.shooting_team = None
                    self.shot_cooldown = 40
                elif by > hy2 + 150:
                    self.shot_active = False
                    self.shooting_team = None

    def save_game_data(self, fps, total_frames):
        # Calculate duration
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # Calculate percentages
        total_poss_frames = self.stats['TeamA']['PossessionFrames'] + self.stats['TeamB']['PossessionFrames']
        poss_a = (self.stats['TeamA']['PossessionFrames'] / total_poss_frames * 100) if total_poss_frames > 0 else 0
        poss_b = (self.stats['TeamB']['PossessionFrames'] / total_poss_frames * 100) if total_poss_frames > 0 else 0

        output_str = (
            "========================================\n"
            "       BASKETBALL ANALYTICS REPORT      \n"
            "========================================\n"
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Video File: {VIDEO_PATH}\n"
            f"Total Frames: {self.frame_count}/{total_frames}\n"
            f"Video Duration: {duration_sec:.2f} seconds\n"
            "----------------------------------------\n"
            "TEAM A (Blue)\n"
            f"  - Possession: {poss_a:.1f}%\n"
            f"  - Field Goals: {self.stats['TeamA']['FGM']}/{self.stats['TeamA']['FGA']}\n"
            "----------------------------------------\n"
            "TEAM B (Red)\n"
            f"  - Possession: {poss_b:.1f}%\n"
            f"  - Field Goals: {self.stats['TeamB']['FGM']}/{self.stats['TeamB']['FGA']}\n"
            "========================================\n"
        )

        print(output_str)
        
        # save the output to a text file
        try:
            with open("game_summary.txt", "w") as f:
                f.write(output_str)
            print("Results saved to 'game_summary.txt'")
        except Exception as e:
            print(f"Error saving file: {e}")

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("Calibrating Teams & Refs...")
        samples = []
        for _ in range(80): 
            ret, frame = cap.read()
            if not ret: break
            
            r_hoop = self.model_hoop(frame, verbose=False, conf=0.5)[0]
            if len(r_hoop.boxes) > 0:
                best = sorted(r_hoop.boxes, key=lambda x: x.conf[0], reverse=True)[0]
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                self.hoop_bbox = (x1, y1, x2, y2)
            
            r = self.model_player(frame, verbose=False, conf=0.6)[0]
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if self.is_valid_player_detection(x1, y1, x2, y2):
                    samples.append(frame[y1:y2, x1:x2])
        
        if samples:
            self.team_classifier.fit_teams(samples)
            self.teams_initialized = True
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frame_count += 1

            r_hoop, r_players, r_ball = self.detect_objects(frame)

            if len(r_hoop.boxes) > 0:
                best_hoop = sorted(r_hoop.boxes, key=lambda x: x.conf[0], reverse=True)[0]
                x1, y1, x2, y2 = map(int, best_hoop.xyxy[0])
                self.hoop_bbox = (x1, y1, x2, y2)

            raw_ball = self.filter_ball(r_ball, frame.shape)
            if raw_ball: final_ball = self.ball_tracker.update(raw_ball[0], raw_ball[1])
            elif self.ball_tracker.missed_frames < 8: final_ball = self.ball_tracker.predict()
            else: final_ball = None; self.ball_history.clear()
            if final_ball: self.ball_history.append(final_ball)

            self.track_players(frame, r_players)
            self.update_logic(final_ball)

            # --- DRAWING ---
            if self.hoop_bbox:
                cv2.rectangle(frame, self.hoop_bbox[:2], self.hoop_bbox[2:], COLOR_HOOP, 2)
                cv2.putText(frame, "HOOP", (self.hoop_bbox[0], self.hoop_bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HOOP, 2)

            for pid, p in self.players.items():
                if p['last_seen'] != self.frame_count: continue
                
                label_text = "Unknown"
                if p['team'] == 'TeamA': c, label_text = COLOR_TEAM_A, "Team A"
                elif p['team'] == 'TeamB': c, label_text = COLOR_TEAM_B, "Team B"
                elif p['team'] == 'Ref': c, label_text = COLOR_REF, "Ref"
                else: c, label_text = (200, 200, 200), "Unknown"
                
                cv2.rectangle(frame, p['bbox'][:2], p['bbox'][2:], c, 2)
                cv2.putText(frame, label_text, (p['bbox'][0], p['bbox'][1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            if final_ball and self.ball_tracker.missed_frames < 5:
                bx, by = final_ball
                
                holder_team, holder_dist = self.find_closest_player_to_ball(final_ball)
                
                if holder_team == "TeamA": ball_color = COLOR_TEAM_A
                elif holder_team == "TeamB": ball_color = COLOR_TEAM_B
                else: ball_color = COLOR_BALL
                
                cv2.rectangle(frame, (bx-12, by-12), (bx+12, by+12), ball_color, 3)
                cv2.putText(frame, "BALL", (bx-15, by-18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
                
                if holder_dist < POSSESSION_DIST_THRESH and holder_team:
                    for pid, p in self.players.items():
                        if p['last_seen'] != self.frame_count: continue
                        if p['team'] == holder_team:
                            dist = math.hypot(p['center'][0] - bx, p['center'][1] - by)
                            if abs(dist - holder_dist) < 5:
                                cv2.line(frame, (bx, by), p['center'], ball_color, 2)
                                break
                
                if len(self.ball_history) > 1:
                    pts = np.array(list(self.ball_history), np.int32)
                    cv2.polylines(frame, [pts], False, COLOR_BALL, 2)

            total = self.stats['TeamA']['PossessionFrames'] + self.stats['TeamB']['PossessionFrames']
            pA = (self.stats['TeamA']['PossessionFrames']/total*100) if total > 0 else 50
            pB = 100 - pA
            
            cv2.rectangle(frame, (0,0), (400, 130), (0,0,0), -1)
            cv2.putText(frame, f"Team A (BLUE): {pA:.1f}% | FG {self.stats['TeamA']['FGM']}/{self.stats['TeamA']['FGA']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEAM_A, 2)
            cv2.putText(frame, f"Team B (RED):  {pB:.1f}% | FG {self.stats['TeamB']['FGM']}/{self.stats['TeamB']['FGA']}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEAM_B, 2)
            
            status = f"SHOT: {self.shooting_team}" if self.shot_active else "LIVE"
            cv2.putText(frame, f"Status: {status} | Poss: {self.current_possession}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Basketball Analytics', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # --- End of Game Summary ---
        self.save_game_data(fps, total_frames)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = BasketballAnalytics()
    system.run()