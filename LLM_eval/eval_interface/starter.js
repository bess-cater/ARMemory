document.getElementById('userForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Get the input value
    const userId = document.getElementById('userId').value;

    // Store the input value in localStorage
    localStorage.setItem('userId', userId);

    // Navigate to the next page
    window.location.href = 'display.html';
});