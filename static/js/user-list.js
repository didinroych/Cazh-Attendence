// ===============================================
// USER LIST JAVASCRIPT
// ===============================================

// DOM Elements
const userList = document.getElementById("userList");
const refreshBtn = document.getElementById("refreshBtn");

// ===============================================
// INITIALIZATION
// ===============================================

document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    fetchUsers();
    console.log('User List Script loaded');
});

function setupEventListeners() {
    refreshBtn.addEventListener('click', fetchUsers);
}

// ===============================================
// USER DATA FUNCTIONS
// ===============================================

async function fetchUsers() {
    showLoading();

    try {
        const response = await fetch("/get-user-face");
        const data = await response.json();

        if (response.ok) {
            displayUsers(data.users || []);
        } else {
            showError("Failed to load user data");
        }
    } catch (error) {
        console.error('Error fetching users:', error);
        showError("Network error. Please check your connection.");
    }
}

function displayUsers(users) {
    userList.innerHTML = "";

    if (users.length === 0) {
        showEmptyState();
        return;
    }

    users.forEach(user => {
        const userCard = createUserCard(user);
        userList.appendChild(userCard);
    });
}

function createUserCard(user) {
    const userCard = document.createElement("div");
    userCard.className = "user-card";

    // Handle both old format (string) and new format (object)
    let name, uid;

    if (typeof user === 'string') {
        // Old format: just name
        name = user;
        uid = 'N/A';
    } else {
        // New format: object with name and uid_face
        name = user.name || 'Unknown';
        uid = user.uid_face || 'N/A';
    }

    userCard.innerHTML = `
        <div class="user-info">
            <div>
                <div class="user-name">${escapeHtml(name)}</div>
                <div class="user-id">ID: ${uid}</div>
            </div>
            <div class="user-actions">
                <button class="refresh-btn" onclick="viewUserDetails('${uid}', '${escapeHtml(name)}')">
                    👁️ View
                </button>
            </div>
        </div>
    `;

    return userCard;
}

function viewUserDetails(uid, name) {
    // For now, just show alert with details
    // In future, could open modal or navigate to detail page
    alert(`User Details:\nName: ${name}\nUID: ${uid}`);
}

// ===============================================
// UI STATE FUNCTIONS
// ===============================================

function showLoading() {
    userList.innerHTML = `
        <div class="user-card loading">
            <div class="user-info">
                <span>⏳ Loading users...</span>
            </div>
        </div>
    `;
}

function showEmptyState() {
    userList.innerHTML = `
        <div class="empty-state">
            <h3>📭 No Users Registered</h3>
            <p>No faces have been registered yet.</p>
            <a href="/regist-new-face" style="color: var(--primary); text-decoration: none; font-weight: 500;">
                ➕ Register your first face
            </a>
        </div>
    `;
}

function showError(message) {
    userList.innerHTML = `
        <div class="user-card" style="border-color: var(--danger); background-color: #fee2e2;">
            <div class="user-info">
                <div style="color: var(--danger);">
                    <strong>❌ Error</strong><br>
                    ${escapeHtml(message)}
                </div>
                <button class="refresh-btn" onclick="fetchUsers()">
                    🔄 Retry
                </button>
            </div>
        </div>
    `;
}

// ===============================================
// UTILITY FUNCTIONS
// ===============================================

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
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