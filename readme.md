#### Article accepted by Frontiers in Robotics and AI
### A Framework for Automatic Behavior Generation in Multi-Function Swarms
Authors: **S. Engebråten**, J. Moen, O. Yakimenko and K. Glette (sondre.engebraten (a) ffi.no)

Multi-function swarms are swarms that solve multiple tasks at once. For example, a quadcopter swarm could be tasked with exploring an area of interest while simultaneously functioning as ad-hoc relays. With this type of multi-function comes the challenge of handling potentially conflicting requirements simultaneously. Using the Quality-Diversity algorithm MAP-elites in combination with a suitable controller structure, a framework for automatic behavior generation in multi-function swarms is proposed. The framework is tested on a scenario with three simultaneous tasks: exploration, communication network creation and geolocation of Radio Frequency (RF) emitters. A repertoire is evolved, consisting of a wide range of controllers, or behavior primitives, with different characteristics and trade-offs in the different tasks. This repertoire enables the swarm to online transition between behaviors featuring different trade-offs of applications depending on the
situational requirements. Furthermore, the effect of noise on the behavior characteristics in MAP-elites is investigated. A moderate number of re-evaluations is found to increase the robustness while keeping the computational requirements relatively low. A few selected controllers are
examined, and the dynamics of transitioning between these controllers are explored. Finally, the study investigates the importance of individual sensor or controller inputs. This is done through ablation, where individual inputs are disabled and their impact on the performance of the swarm controllers is assessed and analyzed. 

The controller used for the swarm is a parametric controller based on Artificial Physics. Artificial Physics is a type of swarm behaviors that use forces acting between agents in order to generate different types of swarm behaviors. The controller for each agent uses eight inputs, the direction and distance to the six nearest neighbors, the direction to the least frequently visited neighboring square and the average predicted radio frequency emitter location. Together they form a velocity setpoint for each agent indicating the direction in which the agent should travel.

<p align="center">
<img src="https://github.com/ForsvaretsForskningsinstitutt/Paper-framework-for-multi-function-swarms/blob/master/controller.png" width=75% height=75%>
</p>

Through adaptation of the parameters of the swarm controller a wide range of different types of behaviors can be evolved. Combining the parametric controller with the quality-diversity method enables the automatic evolution of multi-function swarm behaviors. Below a few selected controllers can be seen:

<p align="center">
<img src="https://github.com/ForsvaretsForskningsinstitutt/Paper-framework-for-multi-function-swarms/blob/master/behaviors.png" >
</p>

Each controller features a different trade-off between the three different application: exploring an area, providing a communication network and localizing radio frequency emitters. Depending on the given scenario the swarm can be adapted by selecting an appropriate behavior from a large repertoire of swarm behaviors.

The full paper can be found at https://www.frontiersin.org/articles/10.3389/frobt.2020.579403/abstract

### Running the provided code

The source code provided assumes Python 2.7. The following libraries and dependencies are required for the source:

    pip install celery deap numpy matplotlib pyyaml redis

To see the visualization of of the combined final repertoire run:

    python mapelites_visualize.py combined.chkpt

The repertoire visualization is interactive and each behavior can be clicked. Each cell in the figure is a behavior with a specific trade-off between the three applications.

Evolving a repertoire of controllers take a long time. The included combined repertoire is the result of about 17000 CPU hours and is the combination of 8 independent evolutionary runs. Even so, the evolutionary process required to evolve one of these repertoires can be started by issuing the command:

	python mapelites_train.py --no_gui

Note that even on a modern machine this would take aproximately 3 months to complete. In total the experiments conducted for this article represents aproximately 17.6 CPU years.

Source code licensed under GLPv3. 

### Citing the article

Paper is cited as:

> Engebråten, Sondre and Moen, Jonas and Yakimenko, Oleg and Glette, Kyrre. (2020, Oct.). A Framework for Automatic Behavior Generation in Multi-Function Swarms. In Frontiers in Robotics and AI.

Bibtex entry as follows:
```
@ARTICLE{engebraaten2020framework,
AUTHOR={Engebråten, Sondre and Moen, Jonas and Yakimenko, Oleg and Glette, Kyrre},   
TITLE={A Framework for Automatic Behavior Generation in Multi-Function Swarms},      
JOURNAL={Frontiers in Robotics and AI},      
YEAR={2020},      
URL={https://www.frontiersin.org/articles/10.3389/frobt.2020.579403/abstract},       
DOI={10.3389/frobt.2020.579403}
}
```

