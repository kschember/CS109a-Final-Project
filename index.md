---
title: Baseball Analysis: Win Probability and the Most Clutch Players
---
 

## Our Question
In this project, we built a model to output the in-game win probability for Major League Baseball games based on situational data such as the score, inning, runners on base, details of the previous play, and other features. We built the model using play-by-play data from baseball-reference.com for every game from the last five years.

When considering the problem we aimed to address with our created model, we sought to find a creative way to rank players. We opted to rank players by “clutchness,” or how well players perform in high leverage situations. Which players consistently have a large impact on their team’s win probability when the situation demands it? Which players create the biggest leaps in win probability when the overall team’s win probability is low?


## Data Overview
Baseball records a huge amount of data that can answer many questions about the sport as well as specific players, but it doesn’t always take into account the context of a play in these statistics. It’s easy to tell which players perform best on a simple scale, but it’s more complicated to determine how players perform when their play matters the most. This is the question we want to answer, and one we think is not obvious based on data already available. Finding the most clutch players is important for obvious reasons: the most valuable players for a team are the ones who can make a real difference in a game when their chance of winning is easily changed. 

This is a challenging question mainly because it requires calculating which plays are high leverage. The idea of a Leverage Index is already defined in the baseball world, but we need to define our own based on the dataset available to us and the win probability model we created. Determining this for baseball is fairly complex, as it fluctuates based on current score, inning, outs left in the inning, runners on base, what batting position the current batter is in the lineup, and more. Out of necessity, we simplified our definition of the Leverage Index to only include some of these variables, but our question of the clutchness of players is not a simple one. 
	
For our EDA work, we graphed a several interactions including the number of wins for home vs. away teams, the average win probability based on the base runner configuration, the win probability mapped with the run differential and batting position, and the win probability mapped with the run differential and inning. All the graphs showed the general trends we expected. We found it interesting that the win probability based on base runner configuration (when the game is tied) differed almost 10% between having bases empty and bases loaded. We also found it compelling and particularly relevant to the type of question we wanted to answer that the probability of winning as compared to run differential and inning was quite dependent on inning. In the first inning, it takes at least a six run differential to be pretty sure a team will win, but by the seventh through ninth innings, it only takes one or two runs to have a good sense. 

We used these findings as evidence that there is solid data to model different situations batters and pitchers are in, but the effects of those situations are not always obvious. Based on these findings, our question of finding the most clutch players seemed appropriate.



## Works Referenced
1. statsbylopez. 2017 March 8. All win probability models are wrong — Some are useful [blog]. [accessed 2017 Dec 6]. https://statsbylopez.com/2017/03/08/all-win-probability-models-are-wrong-some-are-useful/.

2. Watson, Owen. 2015 Oct 9. On Fandom, Leverage, and Emotional Barometers [blog]. [accessed 2017 Dec 6]. https://www.fangraphs.com/blogs/on-fandom-leverage-and-emotional-barometers/.

3. Tango, Tom M. 2006 May 1. Crucial Situations [blog]. [accessed 2017 Dec 6]. https://www.fangraphs.com/tht/crucial-situations/.

Though we were already familiar with the baseball statistics ideas we were considering, Watson’s “On Fandom, Leverage, and Emotional Barometers” was helpful in pointing our direction towards leverage index. The insight into how it can be used and how intriguing it is for both statisticians and sports fans alike was inspiring for our question. Though we couldn’t follow too closely to what statsbylopez offered since that post was about football, we took general information about how to start with our win probability model from that. Finally, Tom Tango’s frequently cited “Crucial Situations” about Leverage Index explained the LI formula from which we modeled our own function. 



## Building the Model
Our implementation of the win probably model required a few important decisions about how we personally wanted the model to predict. After one-hot encoding in EDA, we dropped the redundant and/or unnecessary predictors, we also decided to remove ‘team_at_bat’, ‘batter’, and ‘pitcher’ as predictors. While we could have left the information about teams in the model so it could tell how good a team has been during the past few seasons, we decided this was too much information and that we ultimately wanted a more generalized model. In addition, for our final question we care more about the specific batters and pitchers than a weighting of win probably for overall team performance, and would not that to affect our final result. 

For the WP model, we tried Random Forest, Logistic Regression, LDA, QDA, k-NN,
and AdaBoost. After testing these models on the test set using classification accuracy rate as our performance metric, we realized this metric was not suitable for our goals. We don’t actually care much if our model correctly predicts the final outcome of a game, but rather the incremental win probabilities of every play in a game. Since our final question about clutchness is centered entirely upon play-by-play data, we needed a metric that would explain the success of each model by its mid-game win probabilities. In order to do this, we created a bin accuracy rate with 20 bins that marked success when a WP estimate was within one bin of the result. This way, each WP has to be relatively accurate throughout a game to produce a good R^2. 

Using the bin accuracy rate to cross validate every model, Logistic performed the best, with an R^2 of about 0.97. We expected Logistic to be very good, as it matches logically with baseball statistics and win probabilities. LDA and Random Forest also performed very well on the test set. Interestingly, K-NN performed almost perfectly with classification accuracy, but very poorly with bin accuracy which makes sense given the nature of k-NN. In the end we relied most heavily on the results from the Logistic Regression to answer our clutchness question, but also used LDA and Random Forest to check as they were also valuable models.



## Addressing the Question of Players' Clutchness
We wanted to address the question of whether or not certain players were “more clutch” than others, and therefore we had to determine which situations could be determined as “clutch.” To do this, we used a simplified version of Fangraphs’s (Tom Tango reference above) leverage index. We took every single instance of plays with a certain number of outs remaining in the game (top of the first has 54 outs left, 2 outs in the bottom of the 9th has 1) and run differentials from -10 to 10. In situations where the absolute run differential was more than 10, we simplified and assumed it was 10. Therefore, we had 54*21 different scenarios. From this, we calculated the average absolute win probability change of each of these 1134 states. This average absolute win probability for each of the states was our “leverage index.” To calculate the clutchness of each player, we multiplied the win probability added times the leverage of the play for every single plate appearance in order to give a “clutchness” score for each play. If a player was not clutch, they would have some data points that are massively negative due to underperformance in high leverage situations. We then summed this over all plays.



## Results and Conclusions
Our central WP model was very successful. Based on both our bin accuracy R^2 score and our personal observations comparing the train estimates with what the original data said, all our predicted probabilities were quite accurate. Given more time, it would have been interesting to further explore what effect subset selection would have on our models. We decided that we had a reasonable number of predictors and none of them were strikingly unnecessary, so kept most of them. We also, as mentioned before, made the decision to remove the teams as predictors, which could potentially make a significant difference. Ideally, we would have been able to try a few different subsets we choose on our own and then check it with selections from a stepwise function or other subset selection model. 

Our leverage index model proved very successful in the results. The most clutch batters we found were (in order from most clutch): Paul Goldschmidt, Giancarlo Stanton, Mike Trout, Joey Votto, Miguel Cabrera, David Ortiz, Magglio Ordonez, Barry Bonds, Chipper Jones, Jack Cust, Pat Burrell, Jim Thome, Vladimir Guerrero, Carlos Pena, Rhys Hoskins, Josh Donaldson, and Aaron Boone. 

The most clutch pitchers were (in order from most clutch): Clayton Kershaw, Zack Greinke, Madison Bumgarner, Max Scherzer, Jon Lester, John Lackey, Adam Wainwright, Cole Hamels, Justin Verlander, Johnny Cueto, Chris Sale, Felix Hernandez, Jake Arrieta, Lance Lynn, Jake Peavy, Jacob deGrom, Jeff Samardzija, Jared Weaver, Corey Kluber, Stephen Strasburg, David Price, and Ervin Santana. 

Most of these names are the MLB stars of the past few seasons, confirming that our model is producing reasonable results. Because scraping took so long, we were only able to use data from 2006, 2007, and 2013-2017. Ideally, we would have been able to run this on all the plays for the past ten years. If given more time, we would also continue to adjust our LI model. While it seems to be giving good output now, it could still be far more complex to include more variables or weight the variables differently.

Overall, however, this model successfully predicts win probability, uses that to measure the leverage index of situations, and finally is able to pick out which players perform best under high pressure conditions.


