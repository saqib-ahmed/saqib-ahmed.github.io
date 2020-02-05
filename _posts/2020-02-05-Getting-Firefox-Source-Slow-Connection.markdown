---
layout: post
title: Getting Firefox Source Code on a slow connection
<!-- date:   2020-02-05 15:55:30 +0530 -->
categories: web
---

Firefox is hosted on a mercurial [repository](https://hg.mozilla.org/mozilla-central), and is very large in size (~2GB).
Having tried cloning it multiple times using `hg clone` gave the common `stream ended unexpectedly` or `http connection error`.

Hence, I tried using the bundle method as per the instructions [here](https://developer.mozilla.org/en-US/docs/Mozilla/Developer_guide/Source_Code/Mercurial/Bundles). This time, though I was able to get the code, I was getting the same errors on `hg pull`. 

After searching for answers online, the method which finally worked for me was pulling it chunk by chunk. I wrote the following script to do it for me

```bash
start=0
while :
do
	# Using 3000 as the chunk size
	# Can be adjusted as per your
	times=`expr $start + 3000`
	echo Running $times
	hg pull -r $times https://hg.mozilla.org/mozilla-central
	start=$times
done
```

Once this ends, (you'll get a message saying not enough changesets), you can run the following command
```bash
$ hg update --clean
```

And voila, you have your own firefox repo!