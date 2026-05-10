from ultralytics import YOLO
import cv2

MODEL_PATH = "best.pt"
VIDEO_PATH = "test.mp4"
OUTPUT_PATH = "output_one_swimmer_timer.mp4"

CONF = 0.5
DISTANCE_METER = 50.0

def select_target(frame):
    point = []

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point.clear()
            point.append((x, y))
            print("Selected:", x, y)

    cv2.namedWindow("Select target swimmer")
    cv2.setMouseCallback("Select target swimmer", mouse_click)

    while True:
        temp = frame.copy()
        cv2.putText(temp, "Click target swimmer then press ENTER", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if point:
            cv2.circle(temp, point[0], 8, (0,0,255), -1)

        cv2.imshow("Select target swimmer", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and point:
            break
        if key == ord("q"):
            exit()

    cv2.destroyWindow("Select target swimmer")
    return point[0]

def center_of_box(box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy), (int(x1), int(y1), int(x2), int(y2))

def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("เปิดวิดีโอไม่ได้")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, first_frame = cap.read()
if not ret:
    print("อ่าน frame แรกไม่ได้")
    exit()

target_center = select_target(first_frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

start_frame = None
end_frame = None
final_speed_mps = 0.0
final_speed_kmh = 0.0

print("CONTROL")
print("s = start")
print("e = end")
print("q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    results = model(frame, conf=CONF, verbose=False)
    boxes = results[0].boxes

    annotated = frame.copy()

    if len(boxes) > 0:
        candidates = []

        for box in boxes:
            center, xyxy = center_of_box(box)
            conf = float(box.conf[0].cpu().numpy())
            candidates.append((center, xyxy, conf))

        # เลือกเฉพาะคนที่ใกล้ตำแหน่ง target ล่าสุดที่สุด
        center, xyxy, conf = min(candidates, key=lambda c: dist2(c[0], target_center))
        target_center = center

        x1, y1, x2, y2 = xyxy

        # วาดเฉพาะคนเดียว
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(annotated, center, 5, (0,0,255), -1)
        cv2.putText(annotated, f"Target swimmer {conf:.2f}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ================= TIMER + SPEED =================
    if start_frame is None:
        cv2.putText(annotated, "Press S when swimmer starts", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    elif start_frame is not None and end_frame is None:
        elapsed = (frame_id - start_frame) / fps
        cv2.putText(annotated, f"Time: {elapsed:.2f} s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(annotated, "Press E when swimmer reaches 50m", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    else:
        total_time = (end_frame - start_frame) / fps
        cv2.putText(annotated, f"Total Time: {total_time:.2f} s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(annotated, f"Avg Speed: {final_speed_mps:.2f} m/s", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(annotated, f"Avg Speed: {final_speed_kmh:.2f} km/h", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("One Swimmer Detection + Speed", annotated)
    out.write(annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        start_frame = frame_id
        end_frame = None
        final_speed_mps = 0.0
        final_speed_kmh = 0.0
        print("START")

    elif key == ord("e"):
        if start_frame is not None:
            end_frame = frame_id
            total_time = (end_frame - start_frame) / fps
            final_speed_mps = DISTANCE_METER / total_time
            final_speed_kmh = final_speed_mps * 3.6

            print(f"TIME = {total_time:.2f} sec")
            print(f"SPEED = {final_speed_mps:.2f} m/s")
            print(f"SPEED = {final_speed_kmh:.2f} km/h")

    elif key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved:", OUTPUT_PATH)