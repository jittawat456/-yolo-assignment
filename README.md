# YOLO Animal Detection

เว็บแอปพลิเคชันสำหรับสาธิตการตรวจจับสัตว์ 4 คลาส (**Cat, Cow, Dog, Lion**) ด้วยโมเดล YOLO ที่ฝึกมาแล้ว 2 รูปแบบ คือ Object Detection (Rectangle) และ Instance Segmentation (Polygon) พัฒนาด้วย Flask + Ultralytics YOLO และออกแบบ UI เป็น Light Theme แบบ Responsive

> **Live demo:** <https://yolo-animal-detection-2cpv.onrender.com>
>
> **GitHub:** <https://github.com/jittawat456/-yolo-assignment>
>
> สถานะ: Deploy บน Render สำเร็จ (Free plan) และทดสอบ Rectangle/Polygon บน URL จริงผ่านแล้ว วันที่ 31 ตุลาคม 2026 — หมายเหตุ: Free plan จะ spin down เมื่อไม่มีผู้ใช้งาน และ request แรกอาจใช้เวลาประมาณ 1–2 นาที

---

## 0. ลิงก์เว็บไซต์

| รายการ | ลิงก์ |
|---|---|
| เว็บไซต์ | <https://yolo-animal-detection-2cpv.onrender.com> |
| Health check | <https://yolo-animal-detection-2cpv.onrender.com/health> |
| GitHub Repository | <https://github.com/jittawat456/-yolo-assignment> |
| Render Dashboard | <https://dashboard.render.com/web/srv-dbj14th42hec73bocg6g> |

---

## 1. โมเดลและ Weight ที่ใช้

| โหมด | งาน | Weight | ต้นฉบับ | Task ที่ตรวจพบ |
|---|---|---|---|---|
| Rectangle | Object Detection | `models/box_stage2_best.pt` | `runs/detect/stage2/weights/best.pt` | `detect` |
| Polygon | Instance Segmentation | `models/poly_stage1_best.pt` | `runs/segment/stage1p/weights/best.pt` | `segment` |

ห้ามสลับ Weight ทั้งสองไฟล์ แอปตรวจสอบ `model.task` และรายชื่อคลาสตอนโหลด ถ้า Weight ผิดหรือคลาสไม่ตรง แอปจะหยุดเริ่มทำงานทันทีเพื่อป้องกันการแสดงผลที่ผิด

ค่า SHA-256 ที่ตรวจแล้ว:

```
box_stage2_best.pt   08C69D2BEAC80181F2237FD3065A9E464AF19FBFEA47D69BF437A647583F31BA
poly_stage1_best.pt  3FAE09929D198B1EB148E358D29F6521BBB671DCC82AC545FA20EBD720B5DEFB
```

คลาสและ Class ID อ้างอิงจาก `dataset/classes.txt` (ตาม `dataset/README_split.txt`):

```
0 = cat
1 = cow
2 = dog
3 = lion
```

แอปไม่ได้แก้ Class ID หรือ remap ใด ๆ แต่ตรวจสอบชื่อคลาสจากโมเดลโดยตรง

---

## 2. ฟีเจอร์

- **Dashboard** — ชื่อระบบ, ข้อมูลระบบ, 4 คลาส, จำนวนโมเดลที่โหลดสำเร็จ, ขนาดอัปโหลดสูงสุด
- **Image Upload** — รองรับ JPG/JPEG/PNG, ลากและวาง, Preview ก่อน Predict, จำกัด 10 MB
- **Sample Images** — ภาพตัวอย่าง Cat, Cow, Dog, Lion กดใช้ได้ทันที
- **Rectangle** — Bounding Box พร้อมชื่อคลาสและ Confidence
- **Polygon** — Segmentation Mask/Polygon พร้อมชื่อคลาสและ Confidence
- **Result** — Original Image, Prediction Image, จำนวนวัตถุ, Class, Confidence, Processing Time
- **UX** — Loading ระหว่างประมวลผล, Error ที่อ่านเข้าใจ, Timeout ฝั่ง browser 180 วินาที, ไม่ค้างหน้าเว็บ
- **Responsive** — Desktop, Tablet, Mobile (ทดสอบด้วย Chromium ที่ 1440 / 768 / 390 px)

---

## 3. โครงสร้างโปรเจกต์

```
yolo_webapp_project/
├── app.py                  # Flask app, model loading, validation, prediction API
├── models/
│   ├── box_stage2_best.pt  # Rectangle weight (ห้ามลบ/สลับ)
│   └── poly_stage1_best.pt # Polygon weight (ห้ามลบ/สลับ)
├── static/samples/         # ภาพตัวอย่าง 4 คลาส
├── templates/
│   ├── index.html          # หน้าหลัก (Dashboard + Workspace + Samples + About)
│   └── 404.html
├── tests/test_app.py       # Smoke tests (unittest)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── render.yaml
├── .gitignore
└── README.md
```

---

## 4. รันบน Windows

ต้องใช้ Python 3.11 (ทดสอบด้วย 3.11.9)

```powershell
cd D:\ml_s1\lab\Assignment\yolo_webapp_project

# 1) สร้าง virtual environment
python -m venv .venv

# 2) เปิดใช้งาน (PowerShell)
.\.venv\Scripts\Activate.ps1

# ถ้าเจอ error เรื่อง execution policy ให้รันครั้งเดียว:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3) อัปเกรด pip
python -m pip install --upgrade pip

# 4) ติดตั้ง PyTorch แบบ CPU-only (แนะนำ ลดขนาดดาวน์โหลด)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5) ติดตั้ง dependency ของโปรเจกต์
python -m pip install -r requirements.txt

# 6) เริ่มแอป
python app.py
```

เปิด <http://127.0.0.1:5000>

หมายเหตุ:
- `gunicorn` ถูกติดตั้งเฉพาะ Linux (มี marker ใน `requirements.txt`) เพราะ Gunicorn ไม่รองรับ Windows — บน Windows ใช้ `python app.py`
- ถ้าไม่ติดตั้ง PyTorch ล่วงหน้า `ultralytics` จะพยายามติดตั้ง PyTorch ให้ แต่บน Windows อาจได้ build ที่มี CUDA และมีขนาดใหญ่
- ถ้าต้องการเปลี่ยนพอร์ต: `$env:PORT="8000"; python app.py`

---

## 5. วิธีทดสอบ Rectangle และ Polygon

### 5.1 ผ่านหน้าเว็บ

1. เปิด <http://127.0.0.1:5000>
2. เลือกตัวอย่าง Cat/Cow/Dog/Lion หรืออัปโหลดภาพ JPG/JPEG/PNG (ไม่เกิน 10 MB)
3. ตรวจสอบ Preview และชื่อไฟล์
4. เลือก **Rectangle** → กด **Run prediction** → ดู Bounding Box, Class, Confidence, Processing Time
5. เปลี่ยนเป็น **Polygon** → กด **Run prediction** → ดู Mask/Polygon ทับวัตถุ, Class, Confidence
6. ทดสอบ Error: อัปโหลดไฟล์ `.txt` หรือไฟล์ปลอมที่เปลี่ยนนามสกุลเป็น `.png` ระบบต้องแจ้ง error และไม่ crash

### 5.2 ผ่าน API (PowerShell + curl.exe)

```powershell
# Health check
curl.exe http://127.0.0.1:5000/health

# Rectangle
curl.exe -X POST http://127.0.0.1:5000/predict `
  -F "mode=box" `
  -F "image=@static\samples\randomc_11.jpg" `
  -o result_box.json

# Polygon
curl.exe -X POST http://127.0.0.1:5000/predict `
  -F "mode=poly" `
  -F "image=@static\samples\randoml_100.jpeg" `
  -o result_poly.json

# ไฟล์ผิดประเภท (ต้องได้ HTTP 400)
curl.exe -i -X POST http://127.0.0.1:5000/predict `
  -F "mode=box" `
  -F "image=@README.md"
```

### 5.3 Unit tests

```powershell
python -m unittest discover -s tests -v
```

ชุดทดสอบจะโหลดโมเดลจริงและ predict ภาพตัวอย่างทั้ง 4 ภาพด้วยทั้ง 2 โหมด รวมถึงตรวจไฟล์ผิดประเภท ไฟล์ปลอม ไฟล์เกิน 10 MB และ endpoint `/health`

---

## 6. Build Docker

```powershell
cd D:\ml_s1\lab\Assignment\yolo_webapp_project
docker build -t yolo-animal-detection .
docker run --rm -p 10000:10000 -e PORT=10000 yolo-animal-detection
```

เปิด <http://127.0.0.1:10000> และตรวจ <http://127.0.0.1:10000/health>

รายละเอียดใน Dockerfile:
- ใช้ `python:3.11-slim`
- ติดตั้ง `libgl1`, `libglib2.0-0`, `libgomp1` สำหรับ OpenCV/PyTorch
- ติดตั้ง PyTorch/Torchvision จาก CPU index ก่อน แล้วจึงติดตั้ง `requirements.txt`
- รันด้วย Gunicorn: 1 worker, 4 threads (gthread), timeout 300 วินาที, bind `0.0.0.0:$PORT`
- รันเป็น user ที่ไม่ใช่ root (`appuser`, UID/GID 10001)
- มี `HEALTHCHECK` ไปที่ `/health`

> หมายเหตุ: เครื่องที่ใช้พัฒนาไม่มี Docker จึงไม่ได้รัน `docker build` ในเครื่อง แต่ Render ได้ build Docker image จาก Dockerfile นี้สำเร็จและ service ขึ้นสถานะ `live` แล้ว

---

## 7. ขั้นตอน Deploy บน Render

### 7.1 เตรียม repository

วิธีที่แนะนำ: ให้โฟลเดอร์ `yolo_webapp_project` เป็น **root ของ repository** เพื่อให้ `render.yaml`, `Dockerfile` และ `models/` อยู่ที่ root

```powershell
cd D:\ml_s1\lab\Assignment\yolo_webapp_project
git init
git add .
git commit -m "Prepare YOLO animal detection app for deployment"
git branch -M main
git remote add origin <URL ของ GitHub repository>
git push -u origin main
```

ตรวจสอบว่า `models/box_stage2_best.pt` และ `models/poly_stage1_best.pt` ถูก commit จริง (ไฟล์ละประมาณ 5–7 MB)

### 7.2 สร้าง Web Service

1. เข้า Render Dashboard → **New** → **Blueprint**
2. เลือก GitHub repository ที่ push ไว้
3. Render จะอ่าน `render.yaml` และสร้าง service ชื่อ `yolo-animal-detection`
4. ตรวจสอบ plan: repository นี้ตั้งค่าเป็น **Free (512 MB)** เพราะบัญชีที่ใช้ deploy ไม่มี payment method; หากมีงบประมาณแนะนำเปลี่ยนเป็น **Starter (2 GB RAM)**
5. กด **Apply** / **Create**
6. รอ build และรอ health check ที่ `/health` ผ่าน
7. เปิด URL ที่ Render ให้มา แล้วทดสอบ Rectangle/Polygon ด้วย sample images

Render กำหนดค่า `PORT` ให้อัตโนมัติ แอปอ่านค่าจาก environment variable นี้และ bind ที่ `0.0.0.0` อยู่แล้ว

### 7.3 กรณีใช้ repo ที่มีโฟลเดอร์ซ้อน

Repository ที่ deploy จริงใช้ `yolo_webapp_project` เป็น root ของ repo จึงไม่ต้องตั้ง `dockerContext` เพิ่ม ถ้า push ทั้งโฟลเดอร์ `Assignment` (มี `yolo_webapp_project` อยู่ข้างใน) ต้องแก้ `render.yaml` ที่ root ของ repo เป็น:

```yaml
services:
  - type: web
    name: yolo-animal-detection
    runtime: docker
    plan: free
    region: singapore
    dockerContext: ./yolo_webapp_project
    dockerfilePath: ./yolo_webapp_project/Dockerfile
    healthCheckPath: /health
    autoDeploy: true
```

หรือย้าย/คัดลอก `render.yaml` ไปไว้ที่ root ของ repo พร้อมแก้ path ตามด้านบน

### 7.4 เรื่อง RAM และ Free Plan

วัดบนเครื่องพัฒนาได้ประมาณ **253 MB หลังโหลดโมเดล** และ **396 MB หลัง predict ทั้งสองโหมด** (ยังไม่รวม overhead ของ Gunicorn/OS) บน Render Free plan (512 MB) service ทำงานและ predict ได้จริงในการทดสอบครั้งนี้ แต่เหลือ headroom ไม่มาก

- **Free (512 MB)** — ใช้งานจริงได้ แต่มี cold start นาน (request แรกวัดได้ ~79.6 วินาที) และเสี่ยง OOM หากมี request พร้อมกันหลายรายการ
- **Starter (2 GB)** — แนะนำหากต้องการความเสถียร แต่ต้องเพิ่ม payment method ใน Render

### 7.5 ค่าใช้จ่ายและข้อจำกัดของ Free plan

- Service จะ spin down หลังไม่มีผู้ใช้งานระยะหนึ่ง; request แรกหลังจากนั้นจะช้า
- ไม่มี custom domain แบบ fully managed และอาจมีข้อจำกัดด้านทรัพยากร
- หากต้องการให้อาจารย์เปิดดูได้ทันที ควรเปิด URL ล่วงหน้าอย่างน้อย 1–2 นาทีก่อนนำเสนอ

---

## 8. API

### `GET /`
หน้า Dashboard/Workspace

### `GET /health`
```json
{
  "status": "ok",
  "models_loaded": 2,
  "models_expected": 2,
  "classes": ["cat", "cow", "dog", "lion"]
}
```

### `POST /predict`
`multipart/form-data`

| Field | ค่า |
|---|---|
| `image` | ไฟล์ JPG/JPEG/PNG ไม่เกิน 10 MB |
| `mode` | `box` (Rectangle) หรือ `poly` (Polygon) |

ตัวอย่าง response:

```json
{
  "image": "data:image/jpeg;base64,...",
  "detections": [
    { "index": 1, "class": "cat", "confidence": 94.4, "has_mask": true }
  ],
  "counts": { "cat": 1 },
  "total": 1,
  "mode": "poly",
  "model": "Polygon",
  "task": "Instance Segmentation",
  "stage": "Stage 1",
  "processing_time_ms": 185,
  "inference_time_ms": 120,
  "confidence_threshold": 0.25
}
```

`has_mask` มีเฉพาะโหมด `poly`

---

## 9. ความปลอดภัยและความเป็นส่วนตัว

- ตรวจนามสกุลไฟล์ (`.jpg`, `.jpeg`, `.png`) และตรวจเนื้อหาภาพจริงด้วย Pillow (`verify()` + รูปแบบไฟล์)
- จำกัดขนาดไฟล์ 10 MB ทั้งฝั่ง client และ server
- จำกัดจำนวนพิกเซล 25 ล้านพิกเซล เพื่อป้องกัน decompression bomb
- ไม่ใช้ชื่อไฟล์ของผู้ใช้ในการเขียนลงดิสก์ และไม่บันทึกไฟล์อัปโหลดเลย — ประมวลผลในหน่วยความจำทั้งหมด
- ไม่มีไฟล์ชั่วคราวให้ลบหลังประมวลผล
- Error ที่ส่งกลับ browser เป็นข้อความทั่วไป ไม่มี path ภายในเซิร์ฟเวอร์หรือ stack trace
- ตั้ง `Cache-Control: no-store` กับ `/predict` และ `/health`
- ใส่ security headers พื้นฐาน (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`)
- ไม่มีระบบ login/rate limit เพราะเป็นงานสาธิตทางวิชาการ — หากเปิดเป็นสาธารณะควรเพิ่ม rate limiting

---

## 10. ผลการทดสอบ

สภาพแวดล้อมที่ทดสอบ: Windows, Python 3.11.9, Flask 3.1.3, Ultralytics 8.4.194, PyTorch 2.15.0+cpu, Pillow 13.0.0

| การทดสอบ | ผล |
|---|---|
| Flask เริ่มทำงานและ bind `0.0.0.0` ตาม `PORT` | ผ่าน |
| โหลด Weight ทั้งสองตัว | ผ่าน (`box` = detect, `poly` = segment) |
| ตรวจ Class ID/ชื่อคลาส | ผ่าน (`cat, cow, dog, lion`) |
| Rectangle predict ภาพตัวอย่าง 4 ภาพ | ผ่าน 4/4 ภาพ (Cat/Cow/Dog/Lion) |
| Polygon predict ภาพตัวอย่าง 4 ภาพ | ผ่าน 4/4 ภาพ พร้อม mask |
| อัปโหลดภาพผ่าน HTTP จริง | ผ่าน |
| Sample images และ static files | ผ่าน |
| แสดง Original + Prediction + Class + Confidence + Time | ผ่าน |
| ไฟล์ผิดประเภท / ไฟล์ปลอม / ไฟล์เกิน 10 MB | ผ่าน (HTTP 400/413 พร้อมข้อความเข้าใจง่าย) |
| Unit tests | ผ่าน 13/13 |
| UI tests ด้วย Chromium (Playwright) | ผ่าน 28/28 |
| Responsive 1440 / 768 / 390 px | ผ่าน ไม่มี horizontal overflow |
| RAM | ~253 MB หลังโหลดโมเดล, ~396 MB หลัง predict ทั้งสองโหมด |
| เวลา Predict (CPU) | Rectangle 30–79 ms, Polygon 38–185 ms หลัง warm-up; ครั้งแรก ~1.8–3.0 s |
| Docker build ในเครื่อง | **ไม่ได้รัน** (เครื่องพัฒนาไม่มี Docker) |
| Docker build บน Render | **ผ่าน** — build จาก Dockerfile สำเร็จ |
| Deploy Render จริง | **ผ่าน** — service สถานะ `live`, health check `/health` ตอบ 200 |
| หน้าเว็บจริง (Homepage) | ผ่าน — HTTP 200, แสดงชื่อเว็บและ `2 models ready` |
| Sample images บนเว็บจริง | ผ่าน — HTTP 200, `Content-Type: image/jpeg` |
| Rectangle บนเว็บจริง (Cat) | ผ่าน — ตรวจพบ `cat` 1 ตัว |
| Rectangle บนเว็บจริง (Lion) | ผ่าน — ตรวจพบ `lion` 1 ตัว |
| Polygon บนเว็บจริง (Cat) | ผ่าน — ตรวจพบ `cat` 1 ตัว พร้อม mask |
| Polygon บนเว็บจริง (Lion) | ผ่าน — ตรวจพบ `lion` 1 ตัว พร้อม mask |
| ไฟล์ผิดประเภทบนเว็บจริง | ผ่าน — HTTP 400 พร้อมข้อความเข้าใจง่าย |

เวลา predict measured จากฝั่ง server และจะแตกต่างตาม CPU/ขนาดภาพ

### 10.1 ผลทดสอบบน Render (URL จริง)

ทดสอบเมื่อ 31 ตุลาคม 2026 บน `https://yolo-animal-detection-2cpv.onrender.com` (Free plan, region Singapore)

| โหมด | ภาพ | HTTP | ผลลัพธ์ | เวลารวม (wall clock) | เวลาประมวลผล (server) |
|---|---|---|---|---|---|
| Rectangle | `randomc_11.jpg` (Cat) | 200 | cat × 1 | 79.56 s (cold start) | 78,685 ms |
| Rectangle | `randoml_100.jpeg` (Lion) | 200 | lion × 1 | 3.20 s | 2,709 ms |
| Polygon | `randomc_11.jpg` (Cat) | 200 | cat × 1 + mask | 2.79 s | 2,340 ms |
| Polygon | `randoml_100.jpeg` (Lion) | 200 | lion × 1 + mask | 2.90 s | 2,508 ms |

- Request แรกหลัง service ตื่นจาก spin down ใช้เวลาประมาณ **79.6 วินาที** (cold start)
- Request ถัดไปใช้เวลาประมาณ **2.7–3.2 วินาที**
- ดาวน์โหลดภาพผลลัพธ์และตรวจด้วยสายตาแล้ว พบว่า rectangle และ polygon ตรงกับสัตว์ในภาพจริง

---

## 11. ปัญหาที่ยังเหลือ / ข้อจำกัด

1. **Docker build ในเครื่องยังไม่ได้ทดสอบ** — เครื่องพัฒนาไม่มี Docker แต่ Docker build บน Render สำเร็จและ service ทำงานจริงแล้ว; แนะนำทดสอบ `docker build` ในเครื่องอีกครั้งหากต้องการ reproducibility เต็มรูปแบบ
2. **Deploy บน Free plan (512 MB)** — บัญชี Render ที่ใช้ไม่มี payment method ทำให้ไม่สามารถใช้ Starter plan ได้ จึง deploy ด้วย Free plan; ในการทดสอบนี้ predict ทั้งสองโหมดทำงานได้ แต่ headroom เหลือน้อยและอาจ OOM หากมี request พร้อมกันหลายรายการ
3. **Cold start ของ Free plan** — service จะ spin down เมื่อไม่มีผู้ใช้งาน และ request แรกหลังตื่นใช้เวลาประมาณ **79.6 วินาที**; ควรเปิด URL ล่วงหน้า 1–2 นาทีก่อนนำเสนอ
4. **CPU inference** — ครั้งแรกช้าเพราะ warm-up (~2–3 วินาที) ครั้งถัดไปเร็วขึ้น; บน Render CPU ช้ากว่าเครื่องพัฒนา (ประมาณ 2.3–2.7 วินาทีต่อภาพ)
5. **Request ไปยังโมเดลเดียวกันถูก serialize ด้วย lock** — ป้องกันปัญหา thread-safety แต่ผู้ใช้หลายคนพร้อมกันอาจต้องรอคิว
6. **ไม่มี authentication / rate limiting** — เหมาะกับงานสาธิต ไม่เหมาะเปิดสาธารณะระยะยาวโดยไม่มีมาตรการเพิ่ม
7. **รองรับเฉพาะ JPG/JPEG/PNG** — ไม่รองรับ HEIC/AVIF/WebP ตามข้อกำหนด
8. **ความแม่นยำขึ้นกับข้อมูลฝึก** — Confidence ไม่ใช่การรับประกันผลลัพธ์ และภาพที่อยู่นอก dataset อาจตรวจจับผิด/ไม่พบ
9. **Gunicorn ใช้ไม่ได้บน Windows** — บน Windows ใช้ Flask development server (`python app.py`); production ใช้ Gunicorn ใน Docker/Linux
10. **ต้องมี payment method หากต้องการ Starter plan** — หากต้องการ RAM 2 GB เพื่อความเสถียร ต้องเพิ่มบัตรใน Render Billing แล้วเปลี่ยน plan เป็น `starter`

---

## 12. ไฟล์ที่แก้ไข/เพิ่มในรอบนี้

- `app.py` — ปรับ validation, model task/class checks, processing time, error handling, security headers, health endpoint, thread locks
- `templates/index.html` — ออกแบบ UI ใหม่ทั้งหมด (Light Theme, Dashboard, Workspace, Result, Samples, About, Responsive)
- `templates/404.html` — หน้า 404
- `requirements.txt` — เพิ่ม marker สำหรับ Gunicorn และจำกัด version
- `Dockerfile` — ปรับให้ build/run เป็น non-root, CPU PyTorch, healthcheck, PORT
- `render.yaml` — ตั้งค่า Docker/health check/plan (Free)/env
- `README.md` — เอกสารนี้
- `.gitignore`, `.dockerignore` — ไฟล์ใหม่
- `.gitattributes` — บังคับ LF และกำหนดไฟล์ binary (`.pt`, รูปภาพ)
- `tests/test_app.py` — ชุดทดสอบใหม่
