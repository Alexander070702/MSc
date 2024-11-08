from flask import Flask, render_template, jsonify, request
from gfn.tetris_gfn import TetrisGFlowNet

app = Flask(__name__)
gfn = TetrisGFlowNet()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_trajectory', methods=['POST'])
def generate_trajectory():
    trajectory = gfn.generate_trajectory()
    return jsonify(trajectory)

@app.route('/api/evaluate_state', methods=['POST'])
def evaluate_state():
    state = request.json.get('state')
    reward = gfn.evaluate_state(state)
    return jsonify({'reward': reward})

if __name__ == '__main__':
    app.run(debug=True)
