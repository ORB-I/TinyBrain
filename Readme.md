# Hello, and Welcome!
<p>This, Is TinyBrain. a Neural Network made in a weekend!</p>
<p>using no external libraries aside from NumPy, it's incredibly lightweight (aside from turning my laptop into a jet engine while training. please dont spam the retrain button. I'll have to remove it.), and easy to implement :D</p>

# Self-Hosting TinyBrain
<p>This is the main point of this repo. To self host tinybrain, you need a training file with the following format: Operand, Operand, Operator, Answer.</p>
<h2>THE TRAINING FILE IS NEEDED. TinyBrain WILL NOT OPERATE WITHOUT IT.</h2>
<p>To use TinyBrain, A development server (Flask) has been included, as well as the neccesary endpoints. implementing it should be straight forward. an example site will be provided later if you arent familiar with the REST API.</p>

# Adjusting TinyBrain
<p>This is the main challenge. The training file used to train the public version has around 350 addition examples. the moment your scope increases beyond Addition, your network capacity must increase and training examples must drastically increase as well. Don't be afraid to experiment!</p>
