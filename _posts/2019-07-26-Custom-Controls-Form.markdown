---
layout: post
title: Custom Controls with HTML5 Form in Vue/React
date:   2019-07-26 15:55:30 +0530
categories: web vue react
---

I've seen many people implement custom POST requests for submitting form data while using SPA Frameworks like Vue & React. Most of the times, the need arises because they are using a custom component. The component's data cannot be passed as it is using HTML `<form>`. That's where `input[type=hidden]` comes to the rescue. 

Using the HTML5 Form reduces the hassle of handling the response manually. It handles form resubmission as well. However, it might not be a good choice if your app is entirely an SPA.

Here's an example using hidden input with [vue-select](https://vue-select.org/)

```html
<form action="/api/submitForm">
	<v-select v-model="items" multiple>
	<input name="items" type="hidden" v-model="itemList">
	<!-- auto submit -->
	<button>Send</button>
</form>
```

The javascript code (using vue-cli) would be something like:

```javascript
export default {
	data() {
		return {
			items: []
		}
	},
	
	computed: {
		itemList: function() {
			return items.join(',');
		}
	}
}
```
Let me know if you have any questions below!
