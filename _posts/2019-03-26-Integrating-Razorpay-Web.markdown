---
layout: post
title: Integrating razorpay into your webapp
date:   2019-03-26 15:55:30 +0530
categories: web payments
redirect_from: "/node/web/payments/2019/03/26/Integrating-Razorpay-Web.html"
description: Discussion on how to use razorpay to process payments in your web app using a node.js backend
---
In order to start charging customers we require a payment processor. [Stripe](https://stripe.com) and [braintreepayments.com](http://braintreepayments.com) are popular options in this space. [Razorpay](https://razorpay.com/) is perhaps the best option as of now for the Indian audience.

We require a payment id in order to process a payment using razorpay. We can use the razorpay client side checkout library for this purpose. It is included as:
```html
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
```
Now, the `payment_id` can be generated as follows.

```javascript
const options = {
	key: "YOUR_KEY_ID",
	amount: "2000", // 2000 paise 
	name: "Merchant Name",
	description: "Purchase Description",
	image: "/your_logo.png",
	handler: function (resp) {
		let payment_id = resp.razorpay_payment_id;
		// can be passed to server
		console.log('payment id: ' + payment_id);
	},
	prefill: {
		"name": "John Doe",
		"email": "test@test.com"
	},
	notes: {
		"address": "Hello World"
	},
	theme: {
		"color": "#F37254"
	}
};
const rzp = new Razorpay(options);
// Open Gateway
rzp.open();
```
The `handler` function is called once the payment is done via the checkout form, which includes the payment_id as a parameter and can be used to capture the payment request by the server (by sending an ajax request for example).

As an example, here's how we use node.js on the server along with the `razorpay-node` library to process/capture payment requests.

```javascript
const Razorpay = require('Razorpay');

// Razorpay Instance
const rzp = new Razorpay({
	key_id: '< YOUR KEY ID >',
	key_secret: '< YOUR KEY SECRET >'
});

// Process Payment
function processPayment(authorizedAmount, payment_id) {
	rzp.payments.capture(payment_id, authorizedAmount)	
		.then( (data) => {
			if (data.captured) {
				console.log('Success!');	
			} else {
				console.log('Failed');
			}
		})
};
```
Please note that the amount is specified by both the client and server for safety. You can checkout the excellent [documentation](https://docs.razorpay.com) for more information.

Let me know if you have any questions below!
