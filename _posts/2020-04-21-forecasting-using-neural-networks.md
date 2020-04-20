---
layout: post
title: Implementing a basic Neural Network Part I
description: Implementing a proof-of-concept neural network without any specialized library.
keywords: deep learning, machine learning, AI, neural networks, forecasting, Pytorch, python 
comments: true
---

Neural networks and deep learning has been in vogue for almost the entirety of the last decade and the AI pundits are asserting that this was just a start. AI and DL have opened new ways to explore and exploit the computational prowess of machines. The R&D in the field has mounted to an all-time high in the history of the discipline. 

The rapid research makes it hard for any deep learning framework/approach to stay relevant for long. However, the basics have been established for quite half a century ago. In the early ‘80s, Geoffrey Hinton wrote a paper about backpropagation which is considered to be the most fundamental breakthrough in today’s advancements in artificial intelligence. The idea is considered to be even older than Hinton’s paper, yet, we have a huge room of improvements in both theoretical and practical aspects of deep learning. 

**This blog post series would be mainly focused on scratching the basics of the neural network modeling with regards to implementation without using any specialized deep learning frameworks or libraries.** I'm going to elaborate on the ideas of backpropagation and "learning" in this post with a raw implementation using array libraries. When I first came across neural networks, I found the basic ideas pretty easy to grasp, but I couldn't find any proof-of-concept implementation of those ideas without any buzz words. I would also attempt to provide a basic POC implementation that any novice programmer can understand. 

If you're new to deep learning concepts, I'd urge you to watch this playlist for the simple explanation of the powerful ideas of neural networks. You'd learn the basic jargon of the deep learning which I'm going to use in these posts. This and the following blog posts are complemetary to this playlist. 

<div align="center">
<iframe width="660" height="415" src="https://www.youtube.com/embed/videoseries?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>



### Sample dataset
I'm going to take on a trivial forecasting problem to illustrate the implementation of the concepts. In [this Kaggle dataset](https://www.kaggle.com/marklvl/bike-sharing-dataset), we have the hourly and daily count of rental bikes between the years 2011 and 2012 in the [Capital bike-share system](https://www.capitalbikeshare.com/system-data) in Washington, DC with the corresponding weather and seasonal information. Among many other parameters, this dataset gives temperature, humidity, wind speed, and some day-related parameters for each hour of the day. 

### Sample problem
**To model and train a neural network to harness the information in the dataset to forecast the hourly count of rental bikes for the future.**

### Formulating the solution
To solve the problem, we'll have to devise a neural network structure first. Then we'll implement the feed forward, feedback and the backpropagation mechanism for training. 

#### The Structure
Let's create a class with a constructor to build the structure of the network. We'll create a function to take number of input, hidden and output nodes paramters and the learning rate. I made an obvious choice of having only one hidden layer in this network to spare some complexity. Even this one hidden layer structure works pretty good on the bike-sharing dataset.
```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes     ### 1
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        ######### 2
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes)) 

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes)) # 3
        self.lr = learning_rate 
        
        ######## 3
        self.activation_function = lambda x : (1/(1+np.exp(-x))) 
```

1. These statements initialize the number of input nodes, hidden nodes, and output nodes as passed in the parameters.
2. Initializes the weights from input to the hidden layer. [`np.random.normal`](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html) is creating random values normalized with zero mean and $$\frac{1}{\sqrt{nodes}}$$ standard deviation. To understand the logic behind these choices, you can refer to [this question](https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks).
3. Finally, we have the activation function here. I'm using the sigmoid function as activation: $$\frac{1}{1+ e^{-x}}$$

I'd also like to point out the simplicity of python language here. This code is pretty self explanatory once you know the basic ideas except for maybe the normalization function.


#### Making a prediction
We can make a prediction with our network now. Obviously, without training the network, we'd get some gibberish prediction. With our normal distribution, all the possible outcomes are equally likely without training the network. 

To make a prediction, you get an input and you pass it through the network from all the hidden layers and any activation functions to calculate the output. That's exactly what we're doing in the following code. Pretty self-explanatory. Right?

In mathematical terms, we need to implement: $$\hat{h}=\sigma(W_i.Inputs)$$ and $$\hat{y}=(W_h.\hat{h})$$. We didn't configure any biases in our network for simplicity. 

```python
    def run(self, features):
        # signals into hidden layer obtained by taking dot product
        # of input vector and the input layer weights.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)

        # signals from hidden layer after applying activation function
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # signals into final output layer obtained by taking the dot product
        # of output matrix from hidden layer and the weights from hidden layer
        # to the output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        final_outputs = final_inputs 
        
        return final_outputs
```

This is it for now. I'll write about the backpropagation and training of the network in the next post. See you in the next post. 
