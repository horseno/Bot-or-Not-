Link to the data:

https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data

Links that are useful:

https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/forums/t/14628/share-your-secret-sauce/

What to examine in exploratory analysis:

Bidders vs. # of bids (how many bidders bid once, twice, etc.)
Bidders vs. countries (how many bidders from each country)
Bidders vs. times of day (how many bidders bid at various times of day (per hour, per half hour?))
Bidders vs. category of bid (how many bidders per category)
Bidders vs. closing time of bid (how many bidders bid within 10 seconds of closing time, 1 minute of closing time, etc.)

We can maybe graph these things for all bidders, and then for robots vs. humans (e.g. all bidders vs. number of bids, and then
robots vs. number of bids and humans vs. number of bids). If these histograms are starkly different, maybe that tells us something
about robots' behavior.

Other features:

We can perhaps do the same graphing against these features too:

Median time between a bidder's bid and their most recent bid
Specific weekdays of the bids

These features are drawn from the link above:

- The proportion of each bidder’s bids that were in each day ( I believe there were 9 days of bid data).
- The proportion of each bidder’s bids that were in each of 8 3-hour time slots.
- The mean time of day for the bids of each bidder.
- The total number of bids for each bidder.
- The frequency of each final part of the IP address averaged for each bidder – the first three parts had no positive effect.
- The average proportion of bots associated with each device code. NB – (difficult to put into CV without data leakage – a slightly different data set for each fold!)
- Same as above for each country code.
- The total number of bids made in each auction averaged for each bidder. (i.e. did the bidder tend to bid in popular auctions or less popular ones)
- The maximum number of auctions a bidder has participated in, in one hour.