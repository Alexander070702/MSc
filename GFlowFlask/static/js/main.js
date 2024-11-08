document.getElementById('generate-trajectory').addEventListener('click', () => {
    fetch('/api/generate_trajectory', { method: 'POST' })
        .then(response => response.json())
        .then(trajectory => displayTrajectory(trajectory))
        .catch(error => console.error('Error:', error));
});

function displayTrajectory(trajectory) {
    const board = document.getElementById('tetris-board');
    board.innerHTML = '';  // Clear the board
    // Add visual representation of trajectory (e.g., grid or animation)
    trajectory.forEach(state => {
        const row = document.createElement('div');
        row.className = 'tetris-row';
        state.forEach(cell => {
            const cellDiv = document.createElement('div');
            cellDiv.className = 'tetris-cell';
            cellDiv.style.backgroundColor = cell ? '#333' : '#FFF';
            row.appendChild(cellDiv);
        });
        board.appendChild(row);
    });
}
