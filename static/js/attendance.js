const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('start-btn');
const clockInBtn = document.getElementById('clock-in-btn');
const clockOutBtn = document.getElementById('clock-out-btn');
const stopBtn = document.getElementById('stop-btn');
const statusIndicator = document.getElementById('status-indicator');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');
const liveResult = document.getElementById('live-result');
const resultText = document.getElementById('result-text');
const refreshListBtn = document.getElementById('refresh-list');

// List elements
const currentDateEl = document.getElementById('current-date');
const attendanceListContainer = document.getElementById('attendance-list-container');

// Global variables
let stream = null;

document.addEventListener('DOMContentLoaded', function() {
    updateCurrentDate();
    loadAttendanceList();
    setupEventListeners();
});

function setupEventListeners() {
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    clockInBtn.addEventListener('click', () => performAttendanceAction('clock-in'));
    clockOutBtn.addEventListener('click', () => performAttendanceAction('clock-out'));
    refreshListBtn.addEventListener('click', loadAttendanceList);

    // Auto-stop camera when page is closed/refreshed
    window.addEventListener('beforeunload', () => {
        stopCamera();
    });
}

function updateCurrentDate() {
    const today = new Date();
    const options = {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    };
    currentDateEl.textContent = today.toLocaleDateString('id-ID', options);
}

async function loadAttendanceList() {
    try {
        const today = new Date().toISOString().split('T')[0];
        const response = await fetch(`/attendance-status?date=${today}`);
        const data = await response.json();

        displayAttendanceList(data.attendance || []);
    } catch (error) {
        console.error('Error loading attendance list:', error);
        attendanceListContainer.innerHTML = `
            <div class="empty-state">
                Error loading data
            </div>
        `;
    }
}

function displayAttendanceList(attendanceData) {
    if (attendanceData.length === 0) {
        attendanceListContainer.innerHTML = `
            <div class="empty-state">
                Belum ada yang absen hari ini
            </div>
        `;
        return;
    }

    const listHTML = attendanceData.map(person => {
                const hasClockIn = person.clock_in_time;
                const hasClockOut = person.clock_out_time;

                let statusBadge = '';
                let statusClass = '';

                if (hasClockOut) {
                    statusBadge = 'Selesai';
                    statusClass = 'status-out';
                } else if (hasClockIn) {
                    statusBadge = 'Clock In';
                    statusClass = 'status-in';
                } else {
                    statusBadge = 'Belum Absen';
                    statusClass = 'status-absent';
                }

                return `
            <div class="attendance-item">
                <div class="person-name">${person.name}</div>
                <div class="person-status">
                    <div class="time-info">
                        <div class="time-item">
                            <div class="time-label">Masuk</div>
                            <div class="time-value">${person.clock_in_time || '-'}</div>
                        </div>
                        <div class="time-item">
                            <div class="time-label">Keluar</div>
                            <div class="time-value">${person.clock_out_time || '-'}</div>
                        </div>
                        ${person.duration ? `
                        <div class="time-item">
                            <div class="time-label">Durasi</div>
                            <div class="time-value">${person.duration}</div>
                        </div>
                        ` : ''}
                    </div>
                    <div class="status-badge ${statusClass}">${statusBadge}</div>
                </div>
            </div>
        `;
    }).join('');

    attendanceListContainer.innerHTML = listHTML;
}

// Auto-refresh list every 30 seconds
setInterval(loadAttendanceList, 30000);

async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        // Update UI
        startBtn.disabled = true;
        clockInBtn.disabled = false;
        clockOutBtn.disabled = false;
        stopBtn.disabled = false;
        statusIndicator.textContent = 'Camera Live';
        statusIndicator.className = 'status-indicator status-live';
        hideMessages();

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

        // Update UI
        startBtn.disabled = false;
        clockInBtn.disabled = true;
        clockOutBtn.disabled = true;
        stopBtn.disabled = true;
        statusIndicator.textContent = 'Camera Off';
        statusIndicator.className = 'status-indicator status-stopped';
        liveResult.style.display = 'none';
        video.className = '';
    }
}
async function performAttendanceAction(action) {
    if (!stream) {
        showError('Kamera belum dinyalakan');
        return;
    }

    // Ensure video has proper dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        showError('Video belum siap');
        return;
    }

    // Disable buttons during processing
    clockInBtn.disabled = true;
    clockOutBtn.disabled = true;

    // Capture image from webcam
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
        if (!blob) {
            showError('Gagal mengambil gambar');
            enableButtons();
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'attendance-capture.jpg');

        try {
            const endpoint = action === 'clock-in' ? '/clock-in' : '/clock-out';
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok && data.status === 'success') {
                showSuccess(data.message);
                displayAttendanceResult(data, action);
                loadAttendanceList(); // Refresh the list
            } else {
                showError(data.error || 'Terjadi kesalahan');
            }
        } catch (error) {
            console.error('Attendance error:', error);
            showError('Terjadi kesalahan saat menghubungi server');
        } finally {
            enableButtons();
        }
    }, 'image/jpeg', 0.95);
}

function enableButtons() {
    if (stream) {
        clockInBtn.disabled = false;
        clockOutBtn.disabled = false;
    }
}

function displayAttendanceResult(data, action) {
    const actionText = action === 'clock-in' ? 'Clock In' : 'Clock Out';
    const timeField = action === 'clock-in' ? 'clock_in_time' : 'clock_out_time';
    
    let resultHTML = `
        <div><strong>${actionText} Berhasil!</strong></div>
        <div>Nama: ${data.name}</div>
        <div>Waktu: ${data[timeField]}</div>
    `;
    
    if (action === 'clock-out' && data.duration) {
        resultHTML += `<div>Durasi: ${data.duration}</div>`;
    }
    
    resultText.innerHTML = resultHTML;
    resultText.style.color = '#28a745';
    liveResult.style.display = 'block';
    video.className = 'verified';
}
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 5000);
}

function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        const spans = mobileMenuBtn.querySelectorAll('span');
        spans.forEach(span => span.classList.toggle('active'));
    });
}