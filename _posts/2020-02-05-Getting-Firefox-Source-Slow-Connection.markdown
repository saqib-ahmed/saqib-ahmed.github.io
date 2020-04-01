---
layout: post
title: Getting Firefox Source Code on a slow connection
<!-- date:   2020-02-05 15:55:30 +0530 -->
categories: web
---

Firefox is hosted on a mercurial [repository](https://hg.mozilla.org/mozilla-central), and is very large in size (~2GB).
I tried cloning it multiple times using `hg clone`, and each I time got either a `stream ended unexpectedly` or an `http connection` error.

Hence, I tried using the bundle method as per the instructions [here](https://developer.mozilla.org/en-US/docs/Mozilla/Developer_guide/Source_Code/Mercurial/Bundles). This time, though I was able to get the initial code bundle, I was getting the same errors on pulling further changesets using `hg pull`. 

The method which finally worked for me was pulling the code chunk by chunk. I wrote the following script to do it for me. The chunk size can be modified as per your connection speed.

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

Also note that, if you get the same error while running the script, you can modify the variable `start` to the last chunk which was fetched(it would be printed to your console) and adjust the chunk size accordingly.
