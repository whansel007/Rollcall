# ROLLCALL — On-Device Face Attendance

A browser-based facial recognition attendance system. Enroll each student's face once, then point a phone or laptop camera at the classroom doorway and let students be marked present as they walk in.

- Eliminated manual name-card roll calls, saving TAs an estimated **20 minutes per class** to focus on teaching instead of administration.
- Built a facial recognition system that **marks student attendance from a single scan** via phone or laptop camera at the classroom entrance.

Everything runs **entirely on-device**. Photos, face embeddings and attendance logs never leave the browser — there is no server, no account and no upload.

## Features

- **One-scan attendance** — continuous auto-scan marks whoever is at the door, with a live HUD showing the recognised name.
- **Smart matching** — stores up to 12 face samples per student and matches against the *nearest sample*, which copes with glasses, hair and lighting changes better than a single averaged embedding.
- **Opportunistic learning** — confident live matches are silently added as extra samples, so recognition improves in each room's lighting.
- **Unknown handling** — unrecognised faces are captured with a snapshot and can be enrolled (or dismissed) without leaving the scan screen.
- **Multiple classes** — rosters are stored per class with an optional course code.
- **Three enrolment paths** — bulk photo upload (filename used as a name guess), webcam capture, or inline enrolment from an unknown face.
- **Reports & history** — per-session present/absent/unknown breakdown, with one-click **CSV export** and class **JSON backup/import**.
- **Tunable strictness** — a single slider trades "cannot detect" against the risk of confusing similar faces.

## Tech Stack

| Technology | Role in the project |
| --- | --- |
| **Vanilla JavaScript (ES2020+)** | Entire application logic — routing, state, storage, camera loop, matching. No framework and no build step. |
| **HTML5 / CSS3** | App shell, kiosk UI and the "field instrument" visual theme (custom properties, grid/flex, safe-area insets). |
| **face-api.js** (vendored) | Face detection, landmarks and recognition pipeline. Bundled locally in `vendor/` so the app works offline. |
| **TensorFlow.js** (inside face-api.js) | Neural-net runtime. Uses the **WebGL** backend for speed and falls back to **CPU**; the WASM backend is removed since no `.wasm` binary is shipped. |
| **TinyFaceDetector** | Lightweight single-shot face detector, tuned via `inputSize` / `scoreThreshold` for real-time use. |
| **FaceLandmark68Net** | 68-point facial landmarks, used to align faces before embedding. |
| **FaceRecognitionNet (ResNet-34)** | Produces the 128-dimensional face descriptor (embedding) used for matching. |
| **Euclidean distance** | Squared-Euclidean comparison over 128-d descriptors; a configurable threshold decides known vs. unknown. |
| **IndexedDB** | Local database for classes, students (descriptors + thumbnails) and attendance sessions. |
| **MediaDevices.getUserMedia** | Camera capture from phone or laptop (requires `https://` or `localhost`). |
| **Canvas 2D API** | Draws the mirrored detection overlay (corner brackets + name labels) and crops student thumbnails. |
| **Web Audio API** | Short confirmation/unknown beeps during scanning. |
| **Blob + object URLs** | CSV report and JSON backup downloads. |
| **Google Fonts** | Chakra Petch (display), IBM Plex Sans (body), IBM Plex Mono (data). |

## How It Works

1. **Enrol** — upload a photo, or capture from the webcam. The face is detected, landmarks are computed, and a 128-d descriptor is stored in IndexedDB.
2. **Match** — in the kiosk, every frame is scanned. Each detected face is matched against the nearest stored sample; if the distance is within the threshold, the person is recognised.
3. **Confirm** — a match must repeat across consecutive frames (`CONSEC_HITS`) before attendance is committed, which suppresses false positives.
4. **Record** — marks are timestamped, stored as a session, and can be exported to CSV or reviewed in the history screen.

Key thresholds live in the `CFG` object at the top of `app.js` and can be adjusted from the in-app **Settings** dialog.

## Project Structure

```
index.html           App shell, kiosk and modal markup
app.css              Theme and all component styles
app.js               Application: storage, face engine, enrolment, kiosk, reports
vendor/face-api.js   Bundled face-api.js build (offline)
models/              Face model weights (detector, landmarks, recognition)
```

## Running Locally

No install and no build step are required, but the browser only grants camera access in a **secure context**, so serve the folder over `http://localhost` rather than opening the file directly.

```bash
# any static server works
python -m http.server 8000
# or
npx serve .
```

Then open `http://localhost:8000`.

> Camera access is blocked on plain `file://` and non-HTTPS origins. The app shows an "insecure ctx" badge when this is the case.

## Notes on Privacy & Data

- All data — face descriptors, thumbnails and attendance logs — is stored in the browser's IndexedDB on the device. Nothing is transmitted anywhere.
- Clearing site data deletes the roster, so the **Export backup** JSON is the recommended way to move or protect a class.
- Biometric face templates are used solely for attendance matching and are kept local to the device.
