---
layout: post
title: Implementing a basic Neural Network
description: Implementing a proof-of-concept neural network without any specialized library.
keywords: deep learning, machine learning, AI, neural networks, forecasting, Pytorch, python 
comments: true
---

Neural networks and deep learning has been in vogue for almost the entirety of the last decade and the AI pundits are asserting that this was just a start. AI and DL have opened new ways to explore and exploit the computational prowess of machines. The active research in the field has mounted to an all-time high in the history of the discipline with all the big guns jumping in. 

The rapid research makes it hard for any deep learning framework/approach to stay relevant for long. However, the basics have been established for quite [half a century ago](https://machinelearningknowledge.ai/brief-history-of-deep-learning). Henry J. Kelley in his paper, ["Gradient Theory of Optimal Flight Paths"](https://arc.aiaa.org/doi/abs/10.2514/8.5282) showed the first-ever version of a continuous backpropagation model in 1960 which is basically what we use today at the core of our training mechanism for deep neural networks. 

When I first came across neural networks, I found the basic ideas pretty easy to grasp, but I couldn't find any proof-of-concept implementation of those ideas that can solve some sizeable problem. I'm going to attempt that with this piece. **This blog post would mainly be focused on bridging the gap between the abstract ideas of neural networks and their minimal implementation that can solve some real-world problem** 

I'd urge you to watch this playlist (all four videos) for the visual explanation of the powerful ideas of neural networks by [3blue1brown](https://www.3blue1brown.com/) (if you haven't already). You'd learn the basic jargon of the deep learning which I'm going to use in this post. We'd implement the ideas explained in the playlist.

<div align="center">
<iframe width="660" height="415" src="https://www.youtube.com/embed/videoseries?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>



## Implementation
We're going to implement a neural network without using any specialized deep learning frameworks or libraries like Tensorflow and Pytorch. The libraries do abstract away the inner details for ease of use, but it comes at the cost of obscuring some important details. In my humble opinion, the raw implementation of basic ideas gives a better insight to bridge the gap between concepts and their realization. So here's my take at the implementation.

### The Structure
Let's create a class with a constructor to build the structure of the network. We'll create a function to take the number of input, hidden and output nodes, and the learning rate as parameters. I made an obvious choice of having only one hidden layer in this network to spare some complexity. Even this one hidden layer structure works pretty well for some toy datasets which we'd see in a couple of minutes.

<img src="./assets/generic_neural_network.svg" style="width: 70% !important;">

**Algorithm**
1. Initialize the number of input nodes, hidden nodes, and output nodes.
2. Initialize the weights for all the nodes in all layers. We'll use [`np.random.normal`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html) to create random values normalized with zero mean and $$1/\sqrt{nodes}$$ standard deviation. To understand the logic behind these choices, you can refer to [this question](https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks).
3. Initialitialize the learning rate $$\eta$$
4. Define the activation function. I'm using the sigmoid function as activation: $$1/1+ e^{-x}$$

**Implementation**
```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Step 1
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Step 2
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes)) 

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes)) # 3
        
        # Step 3
        self.lr = learning_rate 
        
        # Step 4
        self.activation_function = lambda x : (1/(1+np.exp(-x))) 
```

I'd also like to point out the simplicity of python language here. This code is pretty self-explanatory once you know the basic ideas except for maybe the normalization function.


### Making a prediction
We can predict with our network now. Obviously, without training the network, we'd get some gibberish predictions. With our normal distribution, all the possible outcomes are equally likely without training the network. 


**Algorithm**

During a forward pass, you take the input and pass it through the network from all the hidden layers and any activation functions to calculate the output. We'll break it down into two steps with hidden layer and output layer predictions. We have only one hidden layer in this network without any activation on the final output layer. We also didn't configure any biases in our network for simplicity.
1. Calculate the output of the hidden layer
  * $$ \hat{h} = \sigma(W_{i-h}\cdot X) $$

2. Calculate the output of the final layer. 
  * $$ \hat y = W_{h-o} \cdot \hat h$$ 

**Implementation**
```python
    def run(self, features):
        
        # Step 1
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # Step 2
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        final_outputs = final_inputs 
        
        return final_outputs
```

### Training the network
The training process consists of the following algorithm. The notations have been borrowed from [Udacity's Nanodegree on deep learning](https://www.udacity.com/course/deep-learning-nanodegree--nd101) and to those of you who can afford it, I would definitely recommend this nanodegree if you want to have a career in deep learning and are confused about where to start.

**Algorithm**
1. Set the weight steps for each layer to zero
  * The input to hidden weights $$\Delta w_{ij} = 0$$
  * The hidden to output weights $$\Delta W_j = 0$$

2. For each record in the training data:
  * Make a forward pass through the network, calculating the output $$\hat y $$
  * Calculate the error gradient in the output unit, $$\delta^o = f'(z)$$ where $$z = \sum_j W_j x_j$$, the input to the output unit.
  * Propagate the errors to the hidden layer $$\delta^h_j = \delta^o W_j f'(h_j)$$
3. Update the weight steps:
  * $$\Delta W_j = \Delta W_j + \delta^o x_j$$
  * $$\Delta w_{ij} = \Delta w_{ij} + \delta^h_j x_i$$
​	 
4. Update the weights, where $$\eta$$ is the learning rate and $$m$$ is the number of records:
  * $$W_j = W_j + \eta \Delta W_j / m$$
  * $$w_{ij} = w_{ij} + \eta \Delta w_{ij} / m$$

**Implementation**

In the implementation, we're gonna modularise the training function.
```python
    def train(self, features, targets):
        n_records = features.shape[0]

        # Step 1
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            # Step 2
            final_outputs, hidden_outputs = self.forward_pass_train(X) 
            
            # Step 3
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, 
                                                                        hidden_outputs, X, y, 
                                                                        delta_weights_i_h, 
                                                                        delta_weights_h_o)

        # Step 4
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
```

#### `self.forward_pass_train(X)`
This function would mostly be the same as the `run()` function from the previous post. All it does is pass the given feature throughout the network to get the prediction. The only difference is, this function should also return the `hidden_ouput` along with the `final_output`.

#### `self.backpropagation()`
Backpropagation is the core of the training process. I'm gonna try to explain it in a bit of detail as well.

**Algorithm**
1. Calculate the cost/error in the final output and intermediate outputs (hidden layer). In this case, we're simply taking the difference between the expected output and the prediction at the final output (for the sake of simplicity). 
  * $$E_{out} = y - \hat{y}$$
  * $$E_{hidden} = W_{hidden-out} \cdot E_{out}$$

2. Calculate the error terms  
  * $$\delta^o_{out} = 1$$ because we don't have any activation function at the output.
  * $$\delta^o_{hidden} = E_{hidden} * \sigma(x)(1 - \sigma(x))$$ (why: [how to calculate the derivative of the sigmoid](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x)).

3. Calculate the weight steps
  * $$\Delta W_{input-hidden} = \Delta W_{input-hidden} + \delta^o_{hidden} * X$$
  * $$\Delta W_{hidden-out} = \Delta W_{hidden-out} + \delta^o_{out} * h_{out}$$

**Implementation**
```python
def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # Step 1
        error = y-final_outputs
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        # Step 2
        output_error_term = error * 1
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Step 3
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_o += (output_error_term * hidden_outputs[:,None])

        return delta_weights_i_h, delta_weights_h_o
```

#### `self.update_weights()`
Finally we update the weights by multiplying the learning rate and the change in the weights.
```python
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records 

        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records 
```

This is it. We have our neural network implemented without any libraries. We have demonstrated the one-to-one correlation between math and the code. 

### Complete code
The complete code of the `NeuralNetwork` class and its usage has been given in [this repository](https://github.com/DevUnleash/Predicting-Bike-Sharing). You can simply execute the Jupyter notebook in the repository to see the neural network in action. I'll write about the experiment in the notebook in some other blog post. This neural network can be used on other datasets. With this much code, you can create "artificial intelligence". 


### References and further readings
1. [3blue1brown Youtube channel](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
2. [Chris Olah's blog](https://colah.github.io/)
3. [Udacity's Deep Learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101)
4. [Michael Nielsen's Book on Deep Learning](http://neuralnetworksanddeeplearning.com/)