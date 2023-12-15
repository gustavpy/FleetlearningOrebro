Fleet Learning

Introduction:
This project started in the beginning of October and has been going on til December. Our mission was to understand the data and visualize it. We were also tasked with finding different strategies to partition the data in different ways to achieve lower loss on the model, and to find things that could affect the training negatively.

First we created a new main code, drawing much inspiration from the initial main code, to enhance the efficiency of our work. By doing this we executed two things at once, it made sure we understood all the segments of the code but also made it possible for us to change and add things that made it easier for us and our work. One significant change was transforming the code into a notebook. This enabled us to quickly test and explore small segments of the code without rerunning everything. Additionally, we have implemented visualizations for training and validation losses, aiding our understanding and evaluation of the strategy's performance over training.

Another improvement is that the code has the ability to stop when overfitting or when the loss stops improving. So if we plan to run 40 rounds and the model stops making progress from the 15th to 20th round, we avoid waiting for the remaining 20 rounds, which take approximately 3 hours to execute with the entire dataset.We have also established the ability to easily adjust global values from a cell labeled 'settings.' This allows us to customize and fine-tune the code without needing to scroll through multiple parts of the script. In the code we also save the parameters. We have then created a code where we can use them for sampletesting and also being able to further train a strategy. This was primarily done to save time, because one whole run with 100% of the data would take around 6-7 hours to complete.

Visualization:
When we were in the beginning of the project we also wanted to be able to get the actual pictures from the dataset. We were able to do this after some work, but we could not select a specific frame to print out, it was just guesswork if you wanted to get a picture from for example London, so we put this mini-project on halt for the time being. But later when we did the did the visualization of the meta data and in particular the density map, we also added so that you could see a particular frame_id when you hover the mouse over a certain location, this meant that we now could get any picture from anywhere in the dataset, which is cool.
Strategies
![Warzawa_density_map](https://github.com/gustavpy/FleetlearningOrebro/assets/149911607/13a988a9-a53c-48d1-8195-fcb52ec36e95)


Strategy 1:
When you train the model you divide the data over several clients or “cars”, one of our missions was to find out the best strategy for dividing the data. leading to the best possible model. 
We have come up with several strategies to improve the model. The strategy that we started with was to just randomize which frames are assigned to a certain Client. I mean it was alright. But we had to find out how other strategies would perform. 

Strategy 2:
The first strategy we tested was to split the data on a certain condition. Along with every frame there is metadata and one example that is annotated is road_condition, if it is “normal”, “wet”, or “snow”. Our first strategy split the data based on that over 40 clients (which is the standard amount of clients). We saw that there are a lot more “normal” road_condition in relation to the others and we wanted to make the clients about the same size, meaning that we created more clients with “normal” road_condition.

Strategy 3:
Another strategy we tried was to split the data according to geographical position based on latitude and longitude that was found in the metadata. This produced a  much better loss then just selecting clients at random and the loss seemed to have a downwards trend even beyond the 40 rounds that we tested it on, as opposed to the random strategy that seemed to stagnate dramatically pretty quickly. To divide the data geographically we have used a KNN-algoritim to divide the data into different clusters based on their position. A possible downside of this is that this did result in quite large differences in Client sizes. This is because a large part of the data is concentrated in cities, in particular the city of Warzaw has approximately 10 000 frames in it. But we are not sure that this is a downside because it might be realistic that some cars do have less data than others, but it would be interesting to know what would happen if we for example had the same concentration of data in kiruna where the road_condition is different. Overall we think that this strategy worked well and is quite realistic, it is also easy to use because it only requires coordinates which are quite easy to get when collecting the data. 
![Screenshot 2023-11-23 095644](https://github.com/gustavpy/FleetlearningOrebro/assets/149911607/cb629433-ca06-40f9-9803-6a5070d6faeb)

Strategy 4:
With our last strategy, we wanted to try something different. We wanted to see what would happen if we took one category of the metadata and then divided the data evenly over all of the clients so that every client was exactly the same size and had exactly the same amount of data. When testing this strategy we used road_conditions which includes “normal”, “wet”, and “snow”. One possible downside of this strategy is that it might not be quite realistic for all clients or “cars” to have the exact same split of the data, but this might indicate that it is good to at least strive for a variety of data assigned to each client. 

When we were in the process of creating this strategy, we discovered that all clients did not get to exactly the same size, the frames in the clients seemed to be a random amount but were approximately 20% smaller than they should have been. After doing some digging around we found out that 20% of the data simply doesn’t work in this code and is regarded as ‘invalid_data’. We decided to look at which data was valid and which was invalid and if we could find some other common factor across the invalid_data apart from it being invalid. We did not find anything like this, which is good because it means that the missing data should not affect our training negatively other than the model training on less data.

We still wanted to try out the original idea of the strategy that was supposed to make all clients have the same amount of frames. To solve this problem, we based the strategy on a copy of ZOD, only containing valid data. This actually resulted in our best Test Loss so far (2.08).

When thinking about imbalances in the dataset which was another mission, we started to think about Great Britain where they drive on the other side of the road. To see if this affected the test loss, we ran the last strategy again but without the Great Brittain data. This resulted in a record-low test loss of 1.89 which seems to indicate that data where they drive on the other side of the road have a negative impact. We believe that if we let this run beyond 40 rounds, the test loss would be even lower.
![Screenshot 2023-12-14 102828](https://github.com/gustavpy/FleetlearningOrebro/assets/149911607/8916bc56-d1ed-47fe-8b81-05f00ea7fd5d)

Another thing to consider with this strategy is the random test/train. If we would have had more time we would have experimented with manually choosing the amount of snow/wet/normal to test with. This should have some impact, negative or positive on the test loss. 

Summary:
If given additional time, we believe we could have enhanced the visualization by individually visualizing each client. So if we were to use the KNN Strategy, we could see each client's images and where we got the data from. That would further give us hindsight of what data gives a better or worse loss. Due to the lack of computational power it took around a whole working day to test one of the strategies, and sometimes the VPN would disconnect while we were running. If we had more time we could have made further progress looking more into different strategies which would have been interesting. We also had an idea of having a database with parameters from a specific location. So if a car is driving in Finland we add the Finland parameters to the general model. Perhaps this could also work on Countries like the UK where there is a drastic difference from other countries.

When we first heard of this project we were all intrigued, but we also felt a little bit intimidated, working with a project that includes self driving cars with fleet léaring, wow. This 3 month journey has been amazing. It started with creating an overview of what needed to be done and understanding the code. We developed a new, comprehensive main code and brainstormed various strategies where we prioritized looking into these strategies. Random, specific road conditions, geographical locations and splitting the data evenly throughout the clients. We also visualized all the data to get a better understanding of how much of each data there is. Due to obstacles we were not able to further try other interesting strategies, but we are still proud of what we accomplished with the time we had.

Finally we want to thank everyone at Zenseact and Volvo who made this project possible, and a special thanks to Viktor, Oscar and Jonas for their inputs, suggestions and supervising throughout the project!





