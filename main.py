from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Базовая конфигурация приложения.
# MVP специально сделан монолитным: вся серверная логика, работа с SQLite,
# mock eGov Digital ID и медицинские правила находятся в одном файле.
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "medcheck.sqlite3"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="mydoctor Platform MVP",
    description="Единая медицинская платформа первичного онлайн-диагностирования",
    version="1.1.0",
)


def render_page(request: Request, template_name: str) -> HTMLResponse:
    """
    Возвращает HTML-страницу с учетом разных версий FastAPI/Starlette.

    В новых версиях Jinja2Templates ожидает request отдельным аргументом,
    а в старых версиях request должен лежать внутри context. Такой небольшой
    адаптер позволяет MVP запускаться и локально, и на бесплатном хостинге.
    """
    try:
        return templates.TemplateResponse(
            request=request,
            name=template_name,
            context={},
        )
    except TypeError:
        return templates.TemplateResponse(template_name, {"request": request})


# -----------------------------------------------------------------------------
# Pydantic-модели входящих запросов.
# -----------------------------------------------------------------------------


class FaceAuthRequest(BaseModel):
    image_data: str = Field(
        ...,
        description="Data URL или base64-строка кадра с камеры",
    )


class VitalsSubmitRequest(BaseModel):
    iin: str
    systolic: int = Field(..., ge=60, le=250)
    diastolic: int = Field(..., ge=40, le=160)
    pulse: int = Field(..., ge=30, le=220)
    promille: float = Field(..., ge=0, le=5)


class DoctorDecisionRequest(BaseModel):
    session_id: int
    decision: str = Field(..., pattern="^(Допущен|Отклонен)$")


class LabSubmitRequest(BaseModel):
    iin: str
    lab_type: str = Field(default="Общий чек-ап")
    hemoglobin: float = Field(..., ge=40, le=220)
    leukocytes: float = Field(..., ge=0, le=100)
    glucose: float = Field(..., ge=0, le=40)
    cholesterol: float = Field(..., ge=0, le=20)
    crp: float = Field(..., ge=0, le=300)


class DiagnosisSubmitRequest(BaseModel):
    iin: str
    complaint: str = Field(..., min_length=3, max_length=800)
    symptoms: list[str] = Field(default_factory=list)
    temperature: float = Field(..., ge=34, le=43)
    duration_days: int = Field(..., ge=0, le=90)
    systolic: int = Field(..., ge=60, le=250)
    diastolic: int = Field(..., ge=40, le=160)
    pulse: int = Field(..., ge=30, le=220)
    oxygen: int = Field(..., ge=50, le=100)
    lab_result_id: Optional[int] = None


class DiagnosisDecisionRequest(BaseModel):
    diagnosis_id: int
    decision: str = Field(
        ...,
        pattern="^(Нужна консультация|Низкий риск|Срочно направить)$",
    )


# -----------------------------------------------------------------------------
# Работа с SQLite.
# -----------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    """Создает подключение к SQLite и возвращает строки как словари."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Создает таблицы и заполняет демо-сотрудников, если база еще пустая."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fio TEXT NOT NULL,
                iin TEXT NOT NULL UNIQUE,
                photo_template TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS medical_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                systolic INTEGER NOT NULL,
                diastolic INTEGER NOT NULL,
                pulse INTEGER NOT NULL,
                promille REAL NOT NULL,
                ai_status TEXT NOT NULL,
                pupil_status TEXT NOT NULL,
                doctor_status TEXT NOT NULL DEFAULT 'Ожидает врача',
                created_at TEXT NOT NULL,
                signature_hash TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fio TEXT NOT NULL,
                iin TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL,
                photo_template TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lab_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                lab_type TEXT NOT NULL,
                hemoglobin REAL NOT NULL,
                leukocytes REAL NOT NULL,
                glucose REAL NOT NULL,
                cholesterol REAL NOT NULL,
                crp REAL NOT NULL,
                ai_summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnosis_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                lab_result_id INTEGER,
                complaint TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                temperature REAL NOT NULL,
                duration_days INTEGER NOT NULL,
                systolic INTEGER NOT NULL,
                diastolic INTEGER NOT NULL,
                pulse INTEGER NOT NULL,
                oxygen INTEGER NOT NULL,
                ai_triage TEXT NOT NULL,
                ai_diagnosis TEXT NOT NULL,
                ai_recommendations TEXT NOT NULL,
                doctor_status TEXT NOT NULL DEFAULT 'Ожидает врача',
                created_at TEXT NOT NULL,
                signature_hash TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (lab_result_id) REFERENCES lab_results(id)
            )
            """
        )

        count = conn.execute("SELECT COUNT(*) AS total FROM employees").fetchone()[
            "total"
        ]
        if count == 0:
            demo_employees = [
                ("Айдос Нурланов", "850412300123", "#1d9bf0"),
                ("Динара Сарсенова", "920730450987", "#12b886"),
                ("Ермек Касымов", "790115350456", "#4263eb"),
            ]
            for fio, iin, color in demo_employees:
                conn.execute(
                    """
                    INSERT INTO employees (fio, iin, photo_template)
                    VALUES (?, ?, ?)
                    """,
                    (fio, iin, create_demo_avatar(fio, color)),
                )
            conn.commit()
            logging.info("Созданы 3 демо-сотрудника для mock eGov Digital ID")

        patients_count = conn.execute(
            "SELECT COUNT(*) AS total FROM patients"
        ).fetchone()["total"]
        if patients_count == 0:
            demo_patients = [
                ("Айгуль Тлеубаева", "940221450321", "+7 701 112 23 45", "#0ea5e9"),
                ("Марат Абишев", "880509300654", "+7 705 445 10 20", "#10b981"),
                ("Сауле Рахметова", "990812450777", "+7 777 830 44 11", "#6366f1"),
            ]
            for fio, iin, phone, color in demo_patients:
                conn.execute(
                    """
                    INSERT INTO patients (fio, iin, phone, photo_template)
                    VALUES (?, ?, ?, ?)
                    """,
                    (fio, iin, phone, create_demo_avatar(fio, color)),
                )
            conn.commit()
            logging.info("Созданы 3 демо-пациента для онлайн-диагностики")


# -----------------------------------------------------------------------------
# Генерация и обработка изображений.
# -----------------------------------------------------------------------------


def create_demo_avatar(fio: str, color: str) -> str:
    """
    Генерирует простое "паспортное" фото сотрудника и сохраняет его как data URL.
    Это удобно для MVP: не нужны внешние файлы и отдельная папка static.
    """
    width, height = 320, 320
    image = Image.new("RGB", (width, height), "#f7fbff")
    draw = ImageDraw.Draw(image)

    # Фон в стиле государственных сервисов: голубой градиент и белая карточка.
    for y in range(height):
        shade = int(245 - y * 0.12)
        draw.line([(0, y), (width, y)], fill=(230, 243, 255 if shade > 220 else 245))
    draw.rounded_rectangle((28, 24, 292, 296), radius=18, fill="#ffffff")

    # Условный портрет: голова, плечи и инициалы.
    draw.ellipse((108, 62, 212, 166), fill=color, outline="#0b3558", width=4)
    draw.pieslice((76, 154, 244, 308), 180, 360, fill="#dbeafe", outline="#0b3558", width=4)
    initials = "".join(part[0] for part in fio.split()[:2]).upper()
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
        small_font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    initials_box = draw.textbbox((0, 0), initials, font=font)
    draw.text(
        ((width - (initials_box[2] - initials_box[0])) / 2, 94),
        initials,
        fill="#ffffff",
        font=font,
    )
    name_box = draw.textbbox((0, 0), fio, font=small_font)
    draw.text(
        ((width - (name_box[2] - name_box[0])) / 2, 260),
        fio,
        fill="#0f172a",
        font=small_font,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def get_pdf_font(size: int) -> ImageFont.ImageFont:
    """
    Подбирает шрифт с поддержкой кириллицы для PDF-заключения.
    На Render обычно есть DejaVu Sans, на macOS - системные шрифты Apple/Arial.
    """
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
) -> int:
    """
    Рисует длинный текст с переносами и возвращает новую Y-координату.
    Это нужно для аккуратного PDF без HTML-рендера и внешних сервисов.
    """
    x, y = xy
    words = str(text or "").replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines or [""]:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line or "A", font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def decode_image_to_array(image_data: str) -> np.ndarray:
    """Декодирует data URL/base64 кадра в RGB-массив NumPy."""
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение") from exc

    return np.array(pil_image)


def image_signature(image_data: str) -> np.ndarray:
    """
    Создает компактную числовую подпись изображения.
    В промышленной версии здесь был бы face embedding, полученный из модели.
    Для MVP используем средние значения каналов и контрастность.
    """
    frame = decode_image_to_array(image_data)
    resized = np.array(Image.fromarray(frame).resize((64, 64)))
    means = resized.mean(axis=(0, 1))
    stds = resized.std(axis=(0, 1))
    return np.concatenate([means, stds])


def mock_check_pupils(frame: np.ndarray) -> str:
    """
    Mock OpenCV-проверки зрачков.
    Функция реально принимает кадр и переводит его в оттенки серого, но вердикт
    для демо фиксированный: интеграционное место под будущую CV-модель готово.
    """
    try:
        # OpenCV подключается лениво: так приложение не падает на старте,
        # если библиотека не установлена или проблемна в конкретной ОС.
        import cv2

        _gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    except Exception:
        # Fallback для MVP: grayscale через NumPy/Pillow, без внешней CV-модели.
        _gray = np.array(Image.fromarray(frame).convert("L"))
    return "Зрачки в норме"


def mock_egov_digital_id(image_data: str) -> dict[str, Any]:
    """
    Имитирует запрос к eGov Digital ID и сопоставление лица с локальной БД.
    Алгоритм выбирает ближайший фото-шаблон по простой пиксельной подписи.
    """
    submitted_signature = image_signature(image_data)

    with get_db() as conn:
        employees = conn.execute("SELECT * FROM employees").fetchall()

    best_employee = None
    best_distance = float("inf")
    for employee in employees:
        template_signature = image_signature(employee["photo_template"])
        distance = float(np.linalg.norm(submitted_signature - template_signature))
        if distance < best_distance:
            best_distance = distance
            best_employee = employee

    if best_employee is None:
        raise HTTPException(status_code=404, detail="Сотрудник не найден")

    # В демо-режиме авторизация проходит, если кадр технически валиден.
    # distance оставляем в ответе как mock confidence для прозрачности.
    confidence = max(0.65, min(0.99, 1 - best_distance / 500))
    return {
        "iin": best_employee["iin"],
        "fio": best_employee["fio"],
        "photo_template": best_employee["photo_template"],
        "confidence": round(confidence, 2),
        "egov_status": "Digital ID подтвержден",
    }


# -----------------------------------------------------------------------------
# Медицинская логика ИИ.
# -----------------------------------------------------------------------------


def analyze_health(vitals: dict[str, Union[float, int]]) -> str:
    """
    Простая rule-based ИИ-оценка.
    В реальном продукте эту функцию можно заменить ML-моделью или сервисом
    медицинского скоринга, не меняя API и фронтенд.
    """
    if float(vitals["promille"]) > 0.0:
        return "Критический риск (Опьянение)"
    if (
        int(vitals["systolic"]) > 140
        or int(vitals["diastolic"]) > 90
        or int(vitals["pulse"]) > 90
    ):
        return "Внимание (Гипертония)"
    return "Норма"


def analyze_lab_results(lab: dict[str, float]) -> str:
    """
    Rule-based интерпретация лабораторных показателей.
    Это не лабораторное заключение, а понятная подсказка для первичного приема.
    """
    flags: list[str] = []
    if lab["hemoglobin"] < 120:
        flags.append("снижен гемоглобин")
    if lab["hemoglobin"] > 170:
        flags.append("повышен гемоглобин")
    if lab["leukocytes"] > 10:
        flags.append("лейкоцитоз, возможное воспаление")
    if lab["leukocytes"] < 4:
        flags.append("снижены лейкоциты")
    if lab["glucose"] >= 7:
        flags.append("повышена глюкоза")
    if lab["cholesterol"] > 5.2:
        flags.append("повышен общий холестерин")
    if lab["crp"] > 10:
        flags.append("повышен C-реактивный белок")

    if not flags:
        return "Анализы без критичных отклонений"
    return "Выявлены отклонения: " + "; ".join(flags)


def analyze_primary_diagnosis(
    request_data: dict[str, Any],
    lab: Optional[dict[str, Any]],
) -> dict[str, str]:
    """
    Mock ИИ первичного диагнозирования.
    Функция собирает жалобу, симптомы, витальные показатели и анализы в один
    предварительный вывод. Финальный диагноз всегда остается за врачом.
    """
    complaint = str(request_data["complaint"]).lower()
    symptoms = [str(symptom).lower() for symptom in request_data.get("symptoms", [])]
    symptom_text = " ".join(symptoms + [complaint])
    recommendations: list[str] = [
        "Это предварительная ИИ-оценка, не заменяющая консультацию врача.",
    ]
    possible_diagnoses: list[str] = []
    triage = "Низкий риск"

    if (
        request_data["oxygen"] < 94
        or request_data["temperature"] >= 39
        or request_data["systolic"] >= 180
        or request_data["diastolic"] >= 120
        or "боль в груди" in symptom_text
        or "одышка" in symptom_text
    ):
        triage = "Высокий риск"
        possible_diagnoses.append("Требуется срочная очная оценка состояния")
        recommendations.append("Обратитесь в неотложную помощь или вызовите скорую.")

    if request_data["temperature"] >= 37.8 and any(
        word in symptom_text for word in ["кашель", "горло", "насморк", "ломота"]
    ):
        triage = "Средний риск" if triage != "Высокий риск" else triage
        possible_diagnoses.append("Вероятная ОРВИ или вирусная инфекция")
        recommendations.append("Контроль температуры, питьевой режим, консультация терапевта.")

    if any(word in symptom_text for word in ["живот", "тошнота", "рвота", "диарея"]):
        triage = "Средний риск" if triage != "Высокий риск" else triage
        possible_diagnoses.append("Возможное нарушение ЖКТ")
        recommendations.append("Оцените обезвоживание и связь с питанием, нужна консультация врача.")

    if request_data["systolic"] > 140 or request_data["diastolic"] > 90:
        triage = "Средний риск" if triage == "Низкий риск" else triage
        possible_diagnoses.append("Повышенное артериальное давление")
        recommendations.append("Повторите измерение давления и покажите врачу динамику.")

    if lab:
        if float(lab["glucose"]) >= 7:
            triage = "Средний риск" if triage == "Низкий риск" else triage
            possible_diagnoses.append("Нарушение углеводного обмена")
            recommendations.append("Рекомендуется повторная глюкоза натощак или HbA1c.")
        if float(lab["crp"]) > 10 or float(lab["leukocytes"]) > 10:
            triage = "Средний риск" if triage == "Низкий риск" else triage
            possible_diagnoses.append("Лабораторные признаки воспаления")
            recommendations.append("Врачу стоит сопоставить CRP/лейкоциты с жалобами.")
        if float(lab["hemoglobin"]) < 120:
            possible_diagnoses.append("Возможный анемический синдром")
            recommendations.append("Проверьте ферритин и обсудите причину снижения гемоглобина.")

    if not possible_diagnoses:
        possible_diagnoses.append("Критичных признаков по анкете не выявлено")
        recommendations.append("Наблюдайте симптомы и обратитесь к врачу при ухудшении.")

    return {
        "triage": triage,
        "diagnosis": "; ".join(dict.fromkeys(possible_diagnoses)),
        "recommendations": " ".join(dict.fromkeys(recommendations)),
    }


def create_signature_hash(session_id: int, decision: str) -> str:
    """Имитирует подписание решения врача через ЭЦП."""
    payload = f"{session_id}:{decision}:{datetime.utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# HTML-страницы.
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def platform_page(request: Request) -> HTMLResponse:
    return render_page(request, "index.html")


@app.get("/terminal", response_class=HTMLResponse, include_in_schema=False)
def terminal_page(request: Request) -> HTMLResponse:
    return render_page(request, "terminal.html")


@app.get("/doctor", response_class=HTMLResponse, include_in_schema=False)
def doctor_page(request: Request) -> HTMLResponse:
    return render_page(request, "doctor.html")


@app.get("/patient", response_class=HTMLResponse, include_in_schema=False)
def patient_page(request: Request) -> HTMLResponse:
    return render_page(request, "patient.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Простой endpoint для проверки доступности сервиса на хостинге."""
    return {"status": "ok", "service": "mydoctor"}


# -----------------------------------------------------------------------------
# API-эндпоинты.
# -----------------------------------------------------------------------------


@app.post("/api/auth/face")
def auth_face(payload: FaceAuthRequest) -> dict[str, Any]:
    """
    Принимает снимок с камеры, имитирует eGov Digital ID и возвращает сотрудника.
    """
    frame = decode_image_to_array(payload.image_data)
    pupil_status = mock_check_pupils(frame)
    employee = mock_egov_digital_id(payload.image_data)
    employee["pupil_status"] = pupil_status
    return employee


@app.post("/api/vitals/submit")
def submit_vitals(payload: VitalsSubmitRequest) -> dict[str, Any]:
    """
    Принимает показатели приборов, оценивает риск и создает сессию для врача.
    """
    with get_db() as conn:
        employee = conn.execute(
            "SELECT * FROM employees WHERE iin = ?",
            (payload.iin,),
        ).fetchone()
        if employee is None:
            raise HTTPException(status_code=404, detail="Сотрудник с таким ИИН не найден")

        vitals = {
            "systolic": payload.systolic,
            "diastolic": payload.diastolic,
            "pulse": payload.pulse,
            "promille": payload.promille,
        }
        ai_status = analyze_health(vitals)

        # Для submit используем mock-статус зрачков без повторной камеры:
        # в реальном терминале статус пришел бы от CV-пакета вместе с кадром.
        pupil_status = "Зрачки в норме"
        cursor = conn.execute(
            """
            INSERT INTO medical_sessions (
                employee_id, systolic, diastolic, pulse, promille,
                ai_status, pupil_status, doctor_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee["id"],
                payload.systolic,
                payload.diastolic,
                payload.pulse,
                payload.promille,
                ai_status,
                pupil_status,
                "Ожидает врача",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()

    return {
        "session_id": cursor.lastrowid,
        "ai_status": ai_status,
        "doctor_status": "Ожидает врача",
        "message": "Данные отправлены врачу",
    }


@app.get("/api/doctor/pending")
def doctor_pending() -> dict[str, list[dict[str, Any]]]:
    """Возвращает все сессии, которые еще ожидают решения врача."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                ms.id,
                ms.systolic,
                ms.diastolic,
                ms.pulse,
                ms.promille,
                ms.ai_status,
                ms.pupil_status,
                ms.doctor_status,
                ms.created_at,
                e.fio,
                e.iin,
                e.photo_template
            FROM medical_sessions ms
            JOIN employees e ON e.id = ms.employee_id
            WHERE ms.doctor_status = 'Ожидает врача'
            ORDER BY ms.created_at DESC
            """
        ).fetchall()

    return {"sessions": [dict(row) for row in rows]}


@app.get("/api/employees/demo")
def demo_employees() -> dict[str, list[dict[str, Any]]]:
    """
    Возвращает демо-сотрудников для терминала.
    Это не замена eGov Digital ID, а страховка для MVP-показа: если браузер
    не отдал веб-камеру, оператор все равно может пройти полный сценарий.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, fio, iin, photo_template
            FROM employees
            ORDER BY id
            """
        ).fetchall()
    return {"employees": [dict(row) for row in rows]}


@app.get("/api/patients/demo")
def demo_patients() -> dict[str, list[dict[str, Any]]]:
    """Возвращает демо-пациентов для кабинета онлайн-диагностики."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, fio, iin, phone, photo_template
            FROM patients
            ORDER BY id
            """
        ).fetchall()
    return {"patients": [dict(row) for row in rows]}


@app.post("/api/labs/submit")
def submit_lab_result(payload: LabSubmitRequest) -> dict[str, Any]:
    """
    Принимает лабораторные показатели и сохраняет их как цифровой анализ.
    В MVP это ручной ввод; в реальности сюда можно подключить ЛИС/лабораторию.
    """
    with get_db() as conn:
        patient = conn.execute(
            "SELECT * FROM patients WHERE iin = ?",
            (payload.iin,),
        ).fetchone()
        if patient is None:
            raise HTTPException(status_code=404, detail="Пациент с таким ИИН не найден")

        lab_values = {
            "hemoglobin": payload.hemoglobin,
            "leukocytes": payload.leukocytes,
            "glucose": payload.glucose,
            "cholesterol": payload.cholesterol,
            "crp": payload.crp,
        }
        ai_summary = analyze_lab_results(lab_values)
        cursor = conn.execute(
            """
            INSERT INTO lab_results (
                patient_id, lab_type, hemoglobin, leukocytes, glucose,
                cholesterol, crp, ai_summary, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient["id"],
                payload.lab_type,
                payload.hemoglobin,
                payload.leukocytes,
                payload.glucose,
                payload.cholesterol,
                payload.crp,
                ai_summary,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()

    return {
        "lab_result_id": cursor.lastrowid,
        "ai_summary": ai_summary,
        "message": "Анализы сохранены в цифровую карту",
    }


@app.get("/api/labs/latest")
def latest_lab_result(iin: str) -> dict[str, Any]:
    """Возвращает последний анализ пациента, если он уже сдавал показатели."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT lr.*
            FROM lab_results lr
            JOIN patients p ON p.id = lr.patient_id
            WHERE p.iin = ?
            ORDER BY lr.created_at DESC
            LIMIT 1
            """,
            (iin,),
        ).fetchone()
    return {"lab": dict(row) if row else None}


def load_person_timeline(iin: str) -> dict[str, Any]:
    """
    Собирает единую медицинскую историю по ИИН.

    Для пациента возвращает анализы и онлайн-диагностику, для сотрудника -
    историю предсменных медосмотров. В реальной платформе эти роли можно
    объединить в одну таблицу пользователей, но для MVP оставляем монолит простым.
    """
    with get_db() as conn:
        patient = conn.execute(
            "SELECT id, fio, iin, phone, photo_template FROM patients WHERE iin = ?",
            (iin,),
        ).fetchone()
        employee = conn.execute(
            "SELECT id, fio, iin, photo_template FROM employees WHERE iin = ?",
            (iin,),
        ).fetchone()

        labs: list[dict[str, Any]] = []
        diagnoses: list[dict[str, Any]] = []
        medical_sessions: list[dict[str, Any]] = []

        if patient:
            labs = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM lab_results
                    WHERE patient_id = ?
                    ORDER BY created_at DESC
                    LIMIT 12
                    """,
                    (patient["id"],),
                ).fetchall()
            ]
            diagnosis_rows = conn.execute(
                """
                SELECT
                    dr.*,
                    lr.lab_type,
                    lr.ai_summary AS lab_summary
                FROM diagnosis_requests dr
                LEFT JOIN lab_results lr ON lr.id = dr.lab_result_id
                WHERE dr.patient_id = ?
                ORDER BY dr.created_at DESC
                LIMIT 12
                """,
                (patient["id"],),
            ).fetchall()
            diagnoses = []
            for row in diagnosis_rows:
                item = dict(row)
                item["symptoms"] = json.loads(item["symptoms"] or "[]")
                diagnoses.append(item)

        if employee:
            medical_sessions = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM medical_sessions
                    WHERE employee_id = ?
                    ORDER BY created_at DESC
                    LIMIT 12
                    """,
                    (employee["id"],),
                ).fetchall()
            ]

    profile = dict(patient) if patient else dict(employee) if employee else None
    if profile is None:
        raise HTTPException(status_code=404, detail="Человек с таким ИИН не найден")
    if patient:
        profile["role"] = "Пациент"
    else:
        profile["role"] = "Сотрудник"
        profile["phone"] = "Не указан"

    return {
        "profile": profile,
        "labs": labs,
        "diagnoses": diagnoses,
        "medical_sessions": medical_sessions,
    }


@app.get("/api/person/{iin}/timeline")
def person_timeline(iin: str) -> dict[str, Any]:
    """Возвращает цифровую карту пациента/сотрудника для кабинета и врача."""
    return load_person_timeline(iin)


def build_conclusion_pdf(timeline: dict[str, Any]) -> bytes:
    """
    Генерирует простое PDF-заключение как изображение.

    Подход выбран специально для MVP: Pillow уже используется в проекте,
    а PDF не зависит от браузера, wkhtmltopdf или платных сервисов.
    """
    profile = timeline["profile"]
    latest_diagnosis = timeline["diagnoses"][0] if timeline["diagnoses"] else None
    latest_lab = timeline["labs"][0] if timeline["labs"] else None
    latest_medical = (
        timeline["medical_sessions"][0] if timeline["medical_sessions"] else None
    )

    width, height = 1240, 1754
    image = Image.new("RGB", (width, height), "#fbfbfd")
    draw = ImageDraw.Draw(image)
    title_font = get_pdf_font(44)
    section_font = get_pdf_font(28)
    bold_font = get_pdf_font(23)
    text_font = get_pdf_font(21)
    small_font = get_pdf_font(17)

    draw.rounded_rectangle((64, 54, width - 64, 214), radius=28, fill="#ffffff")
    draw.text((96, 82), "mydoctor", fill="#0071e3", font=title_font)
    draw.text(
        (96, 142),
        "Цифровое медицинское заключение MVP",
        fill="#1d1d1f",
        font=section_font,
    )
    draw.text(
        (width - 360, 92),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        fill="#6e6e73",
        font=small_font,
    )

    y = 260
    draw.text((82, y), "Пациент / сотрудник", fill="#1d1d1f", font=section_font)
    y += 44
    draw.rounded_rectangle((64, y, width - 64, y + 178), radius=22, fill="#ffffff")
    y += 28
    profile_lines = [
        f"ФИО: {profile['fio']}",
        f"ИИН: {profile['iin']}",
        f"Роль: {profile['role']}",
        f"Телефон: {profile.get('phone', 'Не указан')}",
    ]
    for line in profile_lines:
        draw.text((96, y), line, fill="#1d1d1f", font=text_font)
        y += 36

    y += 42
    draw.text((82, y), "Последнее ИИ-заключение", fill="#1d1d1f", font=section_font)
    y += 44
    draw.rounded_rectangle((64, y, width - 64, y + 370), radius=22, fill="#ffffff")
    y += 28
    if latest_diagnosis:
        draw.text(
            (96, y),
            f"Триаж: {latest_diagnosis['ai_triage']} | Статус врача: {latest_diagnosis['doctor_status']}",
            fill="#0071e3",
            font=bold_font,
        )
        y += 46
        y = draw_wrapped_text(
            draw,
            f"Жалоба: {latest_diagnosis['complaint']}",
            (96, y),
            text_font,
            "#1d1d1f",
            width - 192,
        )
        y = draw_wrapped_text(
            draw,
            f"Предварительно: {latest_diagnosis['ai_diagnosis']}",
            (96, y + 8),
            text_font,
            "#1d1d1f",
            width - 192,
        )
        y = draw_wrapped_text(
            draw,
            f"Рекомендации: {latest_diagnosis['ai_recommendations']}",
            (96, y + 8),
            text_font,
            "#3a3a3c",
            width - 192,
        )
    else:
        draw.text(
            (96, y),
            "Онлайн-диагностика еще не проходилась.",
            fill="#6e6e73",
            font=text_font,
        )
        y += 280

    y = 960
    draw.text((82, y), "Анализы и медосмотры", fill="#1d1d1f", font=section_font)
    y += 44
    draw.rounded_rectangle((64, y, width - 64, y + 330), radius=22, fill="#ffffff")
    y += 28
    if latest_lab:
        draw.text(
            (96, y),
            f"Анализ: {latest_lab['lab_type']} от {latest_lab['created_at']}",
            fill="#1d1d1f",
            font=bold_font,
        )
        y += 44
        lab_text = (
            f"Hb {latest_lab['hemoglobin']}, WBC {latest_lab['leukocytes']}, "
            f"глюкоза {latest_lab['glucose']}, холестерин {latest_lab['cholesterol']}, "
            f"CRP {latest_lab['crp']}. {latest_lab['ai_summary']}"
        )
        y = draw_wrapped_text(draw, lab_text, (96, y), text_font, "#3a3a3c", width - 192)
    else:
        draw.text((96, y), "Лабораторных анализов пока нет.", fill="#6e6e73", font=text_font)
        y += 54

    if latest_medical:
        y += 18
        medical_text = (
            f"Последний медосмотр: АД {latest_medical['systolic']}/"
            f"{latest_medical['diastolic']}, пульс {latest_medical['pulse']}, "
            f"промилле {latest_medical['promille']}. "
            f"ИИ: {latest_medical['ai_status']}. Врач: {latest_medical['doctor_status']}."
        )
        y = draw_wrapped_text(
            draw,
            medical_text,
            (96, y),
            text_font,
            "#3a3a3c",
            width - 192,
        )

    y = 1420
    draw.rounded_rectangle((64, y, width - 64, y + 170), radius=22, fill="#eef6ff")
    y += 28
    disclaimer = (
        "Документ создан в демонстрационной MVP-платформе. ИИ-вывод является "
        "предварительным скринингом и не заменяет очную консультацию врача."
    )
    y = draw_wrapped_text(draw, disclaimer, (96, y), text_font, "#1d1d1f", width - 192)
    draw.text(
        (96, y + 20),
        "Mock ЭЦП: SHA-256 подпись хранится после решения врача.",
        fill="#6e6e73",
        font=small_font,
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=144.0)
    return buffer.getvalue()


@app.get("/api/person/{iin}/conclusion.pdf")
def person_conclusion_pdf(iin: str) -> Response:
    """Формирует PDF-заключение по цифровой карте пациента/сотрудника."""
    timeline = load_person_timeline(iin)
    pdf_bytes = build_conclusion_pdf(timeline)
    filename = f"mydoctor-{iin}-conclusion.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/diagnosis/submit")
def submit_diagnosis(payload: DiagnosisSubmitRequest) -> dict[str, Any]:
    """
    Создает онлайн-обращение первичной диагностики.
    ИИ формирует триаж, возможные направления диагноза и рекомендации врачу.
    """
    with get_db() as conn:
        patient = conn.execute(
            "SELECT * FROM patients WHERE iin = ?",
            (payload.iin,),
        ).fetchone()
        if patient is None:
            raise HTTPException(status_code=404, detail="Пациент с таким ИИН не найден")

        lab = None
        if payload.lab_result_id:
            lab = conn.execute(
                """
                SELECT *
                FROM lab_results
                WHERE id = ? AND patient_id = ?
                """,
                (payload.lab_result_id, patient["id"]),
            ).fetchone()
        else:
            lab = conn.execute(
                """
                SELECT *
                FROM lab_results
                WHERE patient_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (patient["id"],),
            ).fetchone()

        request_data = payload.model_dump()
        ai_result = analyze_primary_diagnosis(request_data, dict(lab) if lab else None)
        cursor = conn.execute(
            """
            INSERT INTO diagnosis_requests (
                patient_id, lab_result_id, complaint, symptoms, temperature,
                duration_days, systolic, diastolic, pulse, oxygen,
                ai_triage, ai_diagnosis, ai_recommendations,
                doctor_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient["id"],
                lab["id"] if lab else None,
                payload.complaint,
                json.dumps(payload.symptoms, ensure_ascii=False),
                payload.temperature,
                payload.duration_days,
                payload.systolic,
                payload.diastolic,
                payload.pulse,
                payload.oxygen,
                ai_result["triage"],
                ai_result["diagnosis"],
                ai_result["recommendations"],
                "Ожидает врача",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()

    return {
        "diagnosis_id": cursor.lastrowid,
        "ai_triage": ai_result["triage"],
        "ai_diagnosis": ai_result["diagnosis"],
        "ai_recommendations": ai_result["recommendations"],
        "message": "Онлайн-обращение отправлено врачу",
    }


@app.get("/api/doctor/diagnosis/pending")
def doctor_diagnosis_pending() -> dict[str, list[dict[str, Any]]]:
    """Возвращает онлайн-обращения, ожидающие врачебного заключения."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                dr.*,
                p.fio,
                p.iin,
                p.phone,
                p.photo_template,
                lr.lab_type,
                lr.ai_summary AS lab_summary,
                lr.hemoglobin,
                lr.leukocytes,
                lr.glucose,
                lr.cholesterol,
                lr.crp
            FROM diagnosis_requests dr
            JOIN patients p ON p.id = dr.patient_id
            LEFT JOIN lab_results lr ON lr.id = dr.lab_result_id
            WHERE dr.doctor_status = 'Ожидает врача'
            ORDER BY
                CASE dr.ai_triage
                    WHEN 'Высокий риск' THEN 1
                    WHEN 'Средний риск' THEN 2
                    ELSE 3
                END,
                dr.created_at DESC
            """
        ).fetchall()

    requests = []
    for row in rows:
        item = dict(row)
        item["symptoms"] = json.loads(item["symptoms"] or "[]")
        requests.append(item)
    return {"requests": requests}


@app.post("/api/doctor/diagnosis/decide")
def doctor_diagnosis_decide(payload: DiagnosisDecisionRequest) -> dict[str, Any]:
    """Сохраняет решение врача по онлайн-диагностике и имитирует ЭЦП."""
    signature_hash = create_signature_hash(payload.diagnosis_id, payload.decision)
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE diagnosis_requests
            SET doctor_status = ?, signature_hash = ?
            WHERE id = ? AND doctor_status = 'Ожидает врача'
            """,
            (payload.decision, signature_hash, payload.diagnosis_id),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Онлайн-обращение не найдено или уже обработано",
        )

    logging.info(
        "ЭЦП врача по онлайн-диагностике: diagnosis_id=%s decision=%s hash=%s",
        payload.diagnosis_id,
        payload.decision,
        signature_hash,
    )
    return {
        "diagnosis_id": payload.diagnosis_id,
        "doctor_status": payload.decision,
        "signature_hash": signature_hash,
        "message": "Заключение врача по онлайн-диагностике подписано ЭЦП",
    }


@app.get("/api/stats")
def platform_stats() -> dict[str, int]:
    """Короткая сводка для главной страницы платформы."""
    today = datetime.now().date().isoformat()
    with get_db() as conn:
        employees = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        pending = conn.execute(
            """
            SELECT COUNT(*)
            FROM medical_sessions
            WHERE doctor_status = 'Ожидает врача'
            """
        ).fetchone()[0]
        completed_today = conn.execute(
            """
            SELECT COUNT(*)
            FROM medical_sessions
            WHERE doctor_status != 'Ожидает врача'
              AND substr(created_at, 1, 10) = ?
            """,
            (today,),
        ).fetchone()[0]
        risk_today = conn.execute(
            """
            SELECT COUNT(*)
            FROM medical_sessions
            WHERE ai_status != 'Норма'
              AND substr(created_at, 1, 10) = ?
            """,
            (today,),
        ).fetchone()[0]
        labs_today = conn.execute(
            """
            SELECT COUNT(*)
            FROM lab_results
            WHERE substr(created_at, 1, 10) = ?
            """,
            (today,),
        ).fetchone()[0]
        diagnosis_pending = conn.execute(
            """
            SELECT COUNT(*)
            FROM diagnosis_requests
            WHERE doctor_status = 'Ожидает врача'
            """
        ).fetchone()[0]

    return {
        "employees": employees,
        "patients": patients,
        "pending": pending,
        "completed_today": completed_today,
        "risk_today": risk_today,
        "labs_today": labs_today,
        "diagnosis_pending": diagnosis_pending,
    }


@app.post("/api/doctor/decide")
def doctor_decide(payload: DoctorDecisionRequest) -> dict[str, Any]:
    """
    Сохраняет решение врача и имитирует ЭЦП через SHA-256-хэш.
    """
    signature_hash = create_signature_hash(payload.session_id, payload.decision)
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE medical_sessions
            SET doctor_status = ?, signature_hash = ?
            WHERE id = ? AND doctor_status = 'Ожидает врача'
            """,
            (payload.decision, signature_hash, payload.session_id),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Сессия не найдена или уже обработана врачом",
        )

    logging.info(
        "ЭЦП врача сгенерирована: session_id=%s decision=%s hash=%s",
        payload.session_id,
        payload.decision,
        signature_hash,
    )
    return {
        "session_id": payload.session_id,
        "doctor_status": payload.decision,
        "signature_hash": signature_hash,
        "message": "Решение врача подписано ЭЦП",
    }


@app.on_event("startup")
def on_startup() -> None:
    init_db()
