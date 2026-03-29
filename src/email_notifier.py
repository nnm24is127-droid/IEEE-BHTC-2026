"""
ParkMod – src/email_notifier.py
Sends violation email notifications to the concerned department.
Attaches evidence image and includes vehicle/plate details.

Uses Python's built-in smtplib (no extra dependencies needed).
Can be configured with any SMTP server (Gmail, Outlook, etc.).
"""

from __future__ import annotations

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("ParkMod.Email")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────
# Email Configuration (update with real credentials)
# ──────────────────────────────────────────────

EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "manyaver288@gmail.com",  # Your sender Gmail
    # ← Gmail App Password (see README)
    "sender_password": "pwsn ygrn lxad scyu",
    "receiver_email": "azmal8881@gmail.com",  # Department / receiver email
    "enabled": True,  # Set True after adding App Password
}


def _build_violation_email(
    vehicle_id: int,
    plate_number: str,
    duration_sec: float,
    location: str,
    timestamp: str,
    evidence_path: Optional[str] = None,
) -> MIMEMultipart:
    """
    Build a rich HTML email with violation details.
    """
    msg = MIMEMultipart("mixed")
    sender = EMAIL_CONFIG.get("sender_email", "")
    receiver = EMAIL_CONFIG.get("receiver_email", "")
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = (
        f"🚨 ParkMod Violation Alert – Vehicle VH-{vehicle_id:04d} | {plate_number}"
    )

    # ── HTML body ──────────────────────────────────────────────────────────
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: #fff;
                    border-radius: 12px; overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

            <!-- Header -->
            <div style="background: linear-gradient(135deg, #1a1a2e, #0f3460);
                        padding: 20px 30px; color: #e94560;">
                <h2 style="margin: 0;">🅿️ ParkMod – Violation Alert</h2>
                <p style="color: #a8b2c1; margin: 5px 0 0;">
                    AI-Based Intelligent Parking Enforcement System
                </p>
            </div>

            <!-- Content -->
            <div style="padding: 25px 30px;">
                <h3 style="color: #e94560; margin-top: 0;">
                    ⚠️ Parking Violation Detected
                </h3>

                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            🚗 Vehicle ID
                        </td>
                        <td style="padding: 10px; color: #555;">
                            VH-{vehicle_id:04d}
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            🔢 Number Plate
                        </td>
                        <td style="padding: 10px; color: #e94560; font-weight: bold;
                                   font-size: 1.1em;">
                            {plate_number}
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            ⏱️ Duration
                        </td>
                        <td style="padding: 10px; color: #555;">
                            {duration_sec:.1f} seconds
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            📍 Location
                        </td>
                        <td style="padding: 10px; color: #555;">
                            {location}
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            🕒 Timestamp
                        </td>
                        <td style="padding: 10px; color: #555;">
                            {timestamp}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold; color: #333;">
                            📋 Status
                        </td>
                        <td style="padding: 10px;">
                            <span style="background: #e94560; color: #fff;
                                         padding: 3px 12px; border-radius: 20px;
                                         font-size: 0.85em; font-weight: bold;">
                                VIOLATION
                            </span>
                        </td>
                    </tr>
                </table>

                <p style="color: #888; font-size: 0.85em; margin-top: 20px;">
                    📎 Evidence image attached below (if available).<br>
                    This is an automated notification from the ParkMod system.
                </p>
            </div>

            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 15px 30px;
                        border-top: 1px solid #eee; text-align: center;">
                <p style="color: #999; font-size: 0.75em; margin: 0;">
                    ParkMod v1.0.0 | AI-Based Intelligent Parking Enforcement System
                    <br>IEEE-BHTC Final-Year Project | 2026
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    # ── Attach evidence image ──────────────────────────────────────────────
    if evidence_path:
        img_path = Path(evidence_path)
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    img_data = f.read()
                img_attachment = MIMEImage(img_data, name=img_path.name)
                img_attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=img_path.name,
                )
                msg.attach(img_attachment)
                logger.info("📎 Evidence image attached: %s", img_path.name)
            except Exception as e:
                logger.warning("Could not attach image: %s", e)

    return msg


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def send_violation_email(
    vehicle_id: int,
    plate_number: str,
    duration_sec: float,
    location: str,
    evidence_path: Optional[str] = None,
) -> dict:
    """
    Send a violation notification email to the concerned department.

    If EMAIL_CONFIG["enabled"] is False, the email is SIMULATED
    (logged but not actually sent). Set enabled=True and fill in
    real SMTP credentials to send real emails.

    Parameters
    ----------
    vehicle_id : int
    plate_number : str
    duration_sec : float
    location : str
    evidence_path : str, optional
        Path to the evidence image file.

    Returns
    -------
    dict : status of the email operation
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = _build_violation_email(
        vehicle_id=vehicle_id,
        plate_number=plate_number,
        duration_sec=duration_sec,
        location=location,
        timestamp=timestamp,
        evidence_path=evidence_path,
    )

    result = {
        "vehicle_id": vehicle_id,
        "plate_number": plate_number,
        "to": EMAIL_CONFIG.get("receiver_email", ""),
        "timestamp": timestamp,
        "evidence": evidence_path or "N/A",
    }

    if not EMAIL_CONFIG["enabled"]:
        # ── SIMULATED (for demo / presentation) ───────────────────────────
        logger.info("=" * 60)
        logger.info("📧 EMAIL NOTIFICATION (SIMULATED)")
        logger.info("   To      : %s", EMAIL_CONFIG.get("receiver_email", ""))
        logger.info("   Subject : %s", msg["Subject"])
        logger.info("   Vehicle : VH-%04d", vehicle_id)
        logger.info("   Plate   : %s", plate_number)
        logger.info("   Duration: %.1fs", duration_sec)
        logger.info("   Location: %s", location)
        logger.info("   Evidence: %s", evidence_path or "None")
        logger.info("   Status  : ✅ SIMULATED – email NOT actually sent")
        logger.info("=" * 60)
        result["status"] = "simulated"
        result["message"] = (
            "Email simulated (set EMAIL_CONFIG['enabled']=True to send real emails)"
        )
        return result

    # ── REAL EMAIL SENDING ─────────────────────────────────────────────────
    try:
        server = smtplib.SMTP(
            EMAIL_CONFIG.get("smtp_server", "smtp.gmail.com"),
            int(EMAIL_CONFIG.get("smtp_port", 587)),
        )
        server.ehlo()
        server.starttls()
        server.login(
            EMAIL_CONFIG.get("sender_email", ""),
            EMAIL_CONFIG.get("sender_password", ""),
        )
        server.send_message(msg)
        server.quit()

        logger.info(
            "✅ Email sent successfully to %s",
            EMAIL_CONFIG.get("receiver_email", ""),
        )
        result["status"] = "sent"
        result["message"] = (
            f"Email sent to {EMAIL_CONFIG.get('receiver_email', '')}"
        )

    except smtplib.SMTPAuthenticationError:
        logger.error("❌ SMTP authentication failed. Check email/password.")
        result["status"] = "auth_error"
        result["message"] = "Authentication failed – check credentials"

    except smtplib.SMTPException as e:
        logger.error("❌ SMTP error: %s", e)
        result["status"] = "smtp_error"
        result["message"] = str(e)

    except Exception as e:
        logger.error("❌ Email error: %s", e)
        result["status"] = "error"
        result["message"] = str(e)

    return result


def send_batch_emails(records: list, location: str) -> list:
    """
    Send email notifications for multiple violations.

    Parameters
    ----------
    records : list of dict
        Each dict: {vehicle_id, plate_number, duration_sec, image_path}
    location : str

    Returns
    -------
    list of result dicts
    """
    results = []
    for rec in records:
        r = send_violation_email(
            vehicle_id=rec.get("vehicle_id", 0),
            plate_number=rec.get("plate_number", "UNKNOWN"),
            duration_sec=rec.get("duration_sec", 0),
            location=location,
            evidence_path=rec.get("image_path", ""),
        )
        results.append(r)
    return results
