import numpy as np
import os
import threading
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.')

SCALE = 1000.0
lr = 0.05

# ─────────────────────────────────────────────
#  NEURAL NETWORK
# ─────────────────────────────────────────────

class TinyBrain:
    def __init__(self, input_size, hidden_layers, output_size, leaky_alpha=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.leaky_alpha = leaky_alpha

        if isinstance(hidden_layers, int):
            self.layer_sizes = [hidden_layers]
        else:
            self.layer_sizes = hidden_layers

        self.num_hidden_layers = len(self.layer_sizes)
        self.weights = []
        self.biases = []

        prev_size = input_size
        for layer_size in self.layer_sizes:
            self.weights.append(np.random.randn(prev_size, layer_size) * np.sqrt(2.0 / prev_size))
            self.biases.append(np.zeros((1, layer_size)))
            prev_size = layer_size

        self.weights_out = np.random.randn(prev_size, output_size) * np.sqrt(2.0 / prev_size)
        self.bias_out = np.zeros((1, output_size))
        self.activations = []

    def leaky_relu(self, x):
        return np.where(x > 0, x, self.leaky_alpha * x)

    def leaky_relu_deriv(self, x):
        return np.where(x > 0, 1.0, self.leaky_alpha)

    def forward(self, x):
        x = np.array(x, dtype=np.float64).reshape(1, -1)
        self.layer_outputs = [x]
        self.layer_raws = []
        self.activations = [x.flatten().tolist()]

        current = x
        for i in range(self.num_hidden_layers):
            raw = np.dot(current, self.weights[i]) + self.biases[i]
            out = self.leaky_relu(raw)
            self.layer_raws.append(raw)
            self.layer_outputs.append(out)
            self.activations.append(out.flatten().tolist())
            current = out

        raw_out = np.dot(current, self.weights_out) + self.bias_out
        self.output_raw = raw_out
        self.output = raw_out
        self.activations.append(raw_out.flatten().tolist())
        return self.output.flatten()

    def train(self, x, y, lr):
        x = np.array(x, dtype=np.float64).reshape(1, -1)
        y = np.array(y, dtype=np.float64).reshape(1, -1)
        self.forward(x.flatten())

        output_error = y - self.output
        output_delta = np.clip(output_error, -0.5, 0.5)

        hidden = self.layer_outputs[-1]
        self.weights_out += lr * np.dot(hidden.T, output_delta)
        self.bias_out += lr * output_delta

        delta = output_delta
        weights = self.weights_out

        for i in range(self.num_hidden_layers - 1, -1, -1):
            raw = self.layer_raws[i]
            prev_out = self.layer_outputs[i]
            error = np.dot(delta, weights.T)
            delta = np.clip(error * self.leaky_relu_deriv(raw), -0.5, 0.5)
            self.weights[i] += lr * np.dot(prev_out.T, delta)
            self.biases[i] += lr * np.sum(delta, axis=0, keepdims=True)
            weights = self.weights[i]

        for w in self.weights + [self.weights_out]:
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                raise ValueError("Weight explosion detected")

        return self.output.flatten()

    def reset(self):
        """Reinitialise all weights — used before retraining."""
        self.__init__(self.input_size, self.layer_sizes, self.output_size, self.leaky_alpha)

    def get_network_state(self):
        all_weight_matrices = self.weights + [self.weights_out]
        return {
            "layer_sizes": [self.input_size] + self.layer_sizes + [self.output_size],
            "activations": self.activations,
            "weights": [w.tolist() for w in all_weight_matrices],
        }


def load_data(filename):
    data = []
    if not os.path.exists(filename):
        return data
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(' ', '').split(',')
            if len(parts) == 4:
                n1, n2 = float(parts[0]), float(parts[1])
                op = parts[2].lower()
                ans = float(parts[3])
                op_map = {'add': 1, 'sub': -1, 'mul': 0}
                if op in op_map:
                    data.append([n1, n2, op_map[op], ans])
    return data


# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────

brain = TinyBrain(3, [3,3], 1)
training_state = {
    "running": False,
    "epoch": 0,
    "total_epochs": 10000,
    "avg_error": None,
    "done": False,
    "fault": False,
    "log": []
}
state_lock = threading.Lock()
brain_lock = threading.Lock()   # Separate lock just for brain access


def run_training():
    global brain
    data = load_data('Example.txt')
    if not data:
        with state_lock:
            training_state["log"].append("Example.txt not found.")
            training_state["fault"] = True
            training_state["done"] = True
            training_state["running"] = False
        return

    with state_lock:
        training_state["log"].append(f"Loaded {len(data)} examples. Training...")

    try:
        for epoch in range(10000):
            total_error = 0
            np.random.shuffle(data)
            for ex in data:
                x = [ex[0] / SCALE, ex[1] / SCALE, ex[2]]
                y = [ex[3] / SCALE]
                with brain_lock:
                    out = brain.train(x, y, lr=0.0001)
                total_error += abs(float(out.flatten()[0]) * SCALE - ex[3])

            avg = total_error / len(data)
            with state_lock:
                training_state["epoch"] = epoch + 1
                training_state["avg_error"] = round(avg, 4)
                if epoch % 500 == 0:
                    training_state["log"].append(f"Epoch {epoch:5d}  error: {avg:.4f}")

    except ValueError as e:
        with state_lock:
            training_state["log"].append(f"FAULT: {e}")
            training_state["fault"] = True
            training_state["done"] = True
            training_state["running"] = False
        return

    with state_lock:
        training_state["log"].append("Training complete!")
        training_state["done"] = True
        training_state["running"] = False

    with brain_lock:
        brain.forward([0.5, 0.5, 1])


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/train', methods=['POST'])
def start_training():
    with state_lock:
        if training_state["running"]:
            return jsonify({"error": "Already training"}), 400
        # Reset everything for a fresh run
        training_state["running"] = True
        training_state["done"] = False
        training_state["fault"] = False
        training_state["epoch"] = 0
        training_state["avg_error"] = None
        training_state["log"] = []

    # Reinitialise weights so we start fresh
    with brain_lock:
        brain.reset()

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route('/api/status')
def status():
    with state_lock:
        return jsonify(dict(training_state))

@app.route('/api/query', methods=['POST'])
def query():
    body = request.json
    expr = body.get('expr', '').strip()

    try:
        if '+' in expr:
            a, b = expr.split('+')
            n1, n2, op, actual = float(a), float(b), 1, float(a)+float(b)
        elif '-' in expr:
            a, b = expr.split('-')
            n1, n2, op, actual = float(a), float(b), -1, float(a)-float(b)
        elif '*' in expr or 'x' in expr:
            expr2 = expr.replace('x','*')
            a, b = expr2.split('*')
            n1, n2, op, actual = float(a), float(b), 0, float(a)*float(b)
        else:
            return jsonify({"parse_error": "Use +, -, or *"}), 400
    except Exception:
        return jsonify({"parse_error": "Couldn't parse — try e.g. 23+41"}), 400

    with brain_lock:
        brain.forward([n1 / SCALE, n2 / SCALE, op])
        out_val = float(brain.output.flatten()[0]) * SCALE
        net_state = brain.get_network_state()

    err = abs(out_val - actual)
    conf = max(0.0, 100.0-(err/abs(actual)*100)) if actual != 0 else (100.0 if err<=0.1 else 0.0)

    return jsonify({
        "result": round(out_val, 2),
        "actual": actual,
        "confidence": round(conf, 1),
        "pred_error": round(err, 3),
        "network": net_state
    })

@app.route('/api/network')
def network_state():
    with brain_lock:
        return jsonify(brain.get_network_state())


if __name__ == '__main__':
    print("TinyBrain server starting at http://localhost:5000")
    training_state["running"] = True
    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    app.run(debug=False, port=5000, host='0.0.0.0')