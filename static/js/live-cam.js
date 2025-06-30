// ===============================================
// LIVE CAMERA FACE VERIFICATION JAVASCRIPT
// ===============================================

// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('start-btn');
const verifyBtn = document.getElementById('verify-btn');
const stopBtn = document.getElementById('stop-btn');
const statusIndicator = document.getElementById('status-indicator');
const autoVerifyIndicator = document.getElementById('auto-verify-indicator');
const errorMessage = document.getElementById('error-message');
const liveResult = document.getElementById('live-result');
const resultText = document.getElementById('result-text');
const autoVerifyCheckbox = document.getElementById('auto-verify-checkbox');
const intervalSlider = document.getElementById('interval-slider');
const intervalValue = document.getElementById('interval-value');
const cameraSelect = document.getElementById('camera-select');
const debugInfo = document.getElementById('debug-info');

// Global variables
let stream = null;
let autoVerifyInterval = null;
let isVerifying = false;
let currentCameraId = null;

// ===============================================
// INITIALIZATION
// ===============================================

document.addEventListener('DOMContentLoaded', function() {
    getCameraList();
    setupEventListeners();
    console.log('Live Camera Script loaded');
});

function setupEventListeners() {
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    verifyBtn.addEventListener('click', () => verifyFace(false));
    autoVerifyCheckbox.addEventListener('change', toggleAutoVerify);
    intervalSlider.addEventListener('input', updateInterval);
    cameraSelect.addEventListener('change', () => {
        if (stream) {
            stopCamera();
            startCamera();
        }
    });

    // Auto-stop camera when page is closed/refreshed
    window.addEventListener('beforeunload', () => {
        stopCamera();
    });
}

// ===============================================
// CAMERA FUNCTIONS
// ===============================================

async function getCameraList() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        if (videoDevices.length === 0) {
            showError('Tidak ada kamera yang terdeteksi');
        }
    } catch (err) {
        console.error('Error getting camera list:', err);
        showError('Gagal mendapatkan daftar kamera');
    }
}

async function startCamera() {
    try {
        const selectedCameraId = cameraSelect.value;
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };

        if (selectedCameraId) {
            constraints.video.deviceId = { exact: selectedCameraId };
        }

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        currentCameraId = selectedCameraId;

        // Update UI
        startBtn.disabled = true;
        verifyBtn.disabled = false;
        stopBtn.disabled = false;
        statusIndicator.textContent = 'Camera Live';
        statusIndicator.className = 'status-indicator status-live';
        errorMessage.style.display = 'none';

        // Re-enumerate devices to get proper labels
        await getCameraList();
        cameraSelect.value = currentCameraId;

    } catch (err) {
        console.error('Error accessing camera:', err);
        showError('Tidak dapat mengakses kamera. Pastikan Anda telah memberikan izin akses kamera.');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;

        // Clear auto-verify if running
        if (autoVerifyInterval) {
            clearInterval(autoVerifyInterval);
            autoVerifyInterval = null;
        }

        // Update UI
        startBtn.disabled = false;
        verifyBtn.disabled = true;
        stopBtn.disabled = true;
        statusIndicator.textContent = 'Camera Off';
        statusIndicator.className = 'status-indicator status-stopped';
        liveResult.style.display = 'none';
        video.className = '';
        autoVerifyCheckbox.checked = false;
        autoVerifyIndicator.textContent = 'Auto-Verify OFF';
        autoVerifyIndicator.style.display = 'none';
    }
}

// ===============================================
// AUTO-VERIFY FUNCTIONS
// ===============================================

function updateInterval() {
    intervalValue.textContent = intervalSlider.value;
    if (autoVerifyInterval) {
        // Restart auto-verify with new interval
        clearInterval(autoVerifyInterval);
        if (autoVerifyCheckbox.checked) {
            startAutoVerify();
        }
    }
}

function startAutoVerify() {
    const intervalSeconds = parseInt(intervalSlider.value);
    autoVerifyInterval = setInterval(() => {
        if (!isVerifying && stream) {
            console.log('Auto-verifying...');
            verifyFace(true);
        }
    }, intervalSeconds * 1000);
    console.log(`Auto-verify started with ${intervalSeconds}s interval`);
}

function toggleAutoVerify() {
    if (autoVerifyCheckbox.checked && stream) {
        // Start auto-verification
        startAutoVerify();
        autoVerifyIndicator.textContent = 'Auto-Verify ON';
        autoVerifyIndicator.style.display = 'block';
        verifyFace(true); // Verify immediately
    } else {
        // Stop auto-verification
        if (autoVerifyInterval) {
            clearInterval(autoVerifyInterval);
            autoVerifyInterval = null;
        }
        autoVerifyIndicator.textContent = 'Auto-Verify OFF';
        autoVerifyIndicator.style.display = autoVerifyCheckbox.checked ? 'block' : 'none';
        liveResult.style.display = 'none';
        video.className = '';
        console.log('Auto-verify stopped');
    }
}

// ===============================================
// FACE VERIFICATION FUNCTIONS
// ===============================================

async function verifyFace(isAuto = false) {
    if (!stream || isVerifying) {
        console.log('Cannot verify: no stream or already verifying');
        return;
    }

    isVerifying = true;
    console.log('Starting verification...');

    // Ensure video has proper dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log('Video not ready yet');
        isVerifying = false;
        return;
    }

    // Capture image from webcam
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convert canvas to blob
    canvas.toBlob(async(blob) => {
        if (!blob) {
            console.error('Failed to create blob');
            isVerifying = false;
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'webcam-capture.jpg');

        try {
            const response = await fetch('/verify-face', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log('Verification result:', data);
            displayResult(data, isAuto);
        } catch (error) {
            console.error('Verification error:', error);
            if (!isAuto) {
                showError('Terjadi kesalahan saat menghubungi server');
            }
        } finally {
            isVerifying = false;
        }
    }, 'image/jpeg', 0.95);
}

function displayResult(data, isAuto = false) {
    if (data.error) {
        if (!isAuto) {
            showError(data.error);
        }
        return;
    }

    if (data.status === 'no face detected') {
        video.className = 'no-face';
        resultText.textContent = 'Wajah tidak terdeteksi';
        resultText.style.color = '#dc3545';
    } else if (data.status === 'ok') {
        const isSpoof = data.spoof;
        video.className = isSpoof ? 'spoof' : 'verified';
        resultText.innerHTML = `
            <div>Nama: ${data.name}</div>
            <div>Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
            <div>UID: ${data.uid_face || 'Unknown'}</div>
            <div>Status: ${isSpoof ? 'Kemungkinan Spoof' : 'Valid'}</div>
        `;
        resultText.style.color = isSpoof ? '#ffc107' : '#28a745';
    }

    liveResult.style.display = 'block';
}

// ===============================================
// MESSAGE FUNCTIONS
// ===============================================

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// ===============================================
// MOBILE MENU TOGGLE
// ===============================================

const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        const spans = mobileMenuBtn.querySelectorAll('span');
        spans.forEach(span => span.classList.toggle('active'));
    });
}