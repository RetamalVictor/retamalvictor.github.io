---
title: "Weight-Space Programming for Drones: Editing PX4's Neural Flight Controller Without Training"
date: "2026-11-06"
tags: ["robotics", "control", "drones", "px4", "llm", "weight-space-programming", "isaac-sim"]
summary: "RAPTOR is the small recurrent controller that ships in PX4. I gave its weights, the vehicle physics and a recording of its behaviour to Claude Opus and asked it to explain what it saw and change the weights so the behaviour changed. No training, no gradients, every edit judged on starts the model never flew. A showcase of what came out, with the caveats attached."
readTime: "6 min"
notebook: ""
featured: true
---

# Weight-Space Programming for Drones: Editing PX4's Neural Flight Controller Without Training

<figure class="wm-fig wm-hero wm-diagram">
 <img src="/images/weight-surgeon/hero_method_v7.svg" alt="Three stations left to right. The original RAPTOR weights as a weight matrix and a climb that overshoots; Claude Opus in the middle, exchanging edits and recordings with a harness; the same weight matrix with one blue cell and a clean climb.">
 <figcaption>One repair, start to finish. Grey is the original RAPTOR weights, the controller PX4 ships. Blue is the same controller with one entry of its GRU input matrix changed, judged on starts the model never flew. There is one set of weights for all three vehicles, so every edit is re-flown on all of them.</figcaption>
</figure>

<figure class="wm-fig wm-strip">
 <div class="wm-strip-row">
  <img src="/images/weight-surgeon/isaac_racer_step_v2.gif" alt="Two racing quads fly a sideways step then a climb; the white-beacon quad overshoots the climb, the blue-beacon quad stops at the target.">
  <img src="/images/weight-surgeon/isaac_racer_course_v3.gif" alt="Two racing quads fly a three-dimensional course through a warehouse; the white-beacon quad lags and cuts corners, the blue-beacon quad holds the line.">
  <img src="/images/weight-surgeon/isaac_racer_flip_v2.gif" alt="Two racing quads released nearly upside down; the white-beacon quad falls and freezes red on the floor, the blue-beacon quad rights itself and climbs back.">
  <img src="/images/weight-surgeon/isaac_racer_track_fast_v3.gif" alt="Two racing quads fly a fast figure-eight; the white-beacon quad settles on a wide orbit that never finds the eight, the blue-beacon quad traces it.">
 </div>
 <figcaption>The racer in Isaac Sim, the original RAPTOR weights (white beacon) against the merged controller (blue beacon), same start and same physics in each pair: a 1 m step, a 3D course, a release near inverted, and the 3.5 s figure-eight.</figcaption>
</figure>

At the start of November, MIT CSAIL published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab), in which a language model repaired a stalled car policy by reading its weights and editing them directly, and did the same for a looping language model. I work with drones and definetly wanted to give it a try to see if this is something I could use to hopefully fly a drone one day. 

So the question here is narrow. Take the weights of a flight controller that has a release tag. Fly it in the simulator it was trained in. Pick a behaviour you can see and describe. Hand the weights, the physics and a recording to a model and ask it to change the weights so the behaviour changes. Then check, on starts the model never saw, whether it did and whether anything else got worse.

## The trained model

[RAPTOR](https://arxiv.org/abs/2509.11481) (Eschmann et al., 2025) is one small recurrent network, a GRU with 16 units and 2,084 weights, trained to fly quadrotors from palm-sized to several kilograms without retuning. PX4 ships it as the experimental `mc_raptor` module. The weights operated on here are [the checkpoint PX4 pins](https://github.com/rl-tools/px4-blob/blob/bc6683a9a2cff29bede2add18c0d7e3ac0cc9c27/policy.tar) for [that module](https://github.com/PX4/PX4-Autopilot/tree/bf59c29e8b8cb9ac66729efc8ce26e41a875f634/src/modules/mc_raptor). Throughout, the original RAPTOR weights are that checkpoint untouched, and the merged controller is the same network after the eleven edits described below, the output of this tool.

<details>
<summary>The network, in one figure</summary>
<figure class="wm-fig wm-diagram">
 <img src="/images/weight-surgeon/network_v3.svg" alt="Pipeline left to right. A column of 22 input cells in five slices: position error, rotation matrix, velocity error, body rates, previous output. Then a dense layer, a 16-unit GRU with a loop arrow carrying its state to the next step, a dense output layer, and four motor command cells under a quad glyph.">
 <figcaption>Dense, GRU, dense: 2,084 weights, stepped at 100 Hz. The GRU state is the only memory, and the previous output is fed back as an input.</figcaption>
</figure>
<ul>
 <li><strong>It sees</strong> its own state relative to the reference: position error clipped to half a metre per axis, attitude as the full rotation matrix, velocity error clipped to a metre per second, body rates straight from the gyro, and its own output from the last step.</li>
 <li><strong>It does not see</strong> absolute position, reference velocity, battery, motor speed, a map or a camera. Every trajectory looks like "where I am relative to where I should be right now".</li>
 <li><strong>It commands motors directly.</strong> One value per motor from minus one to one, which PX4 maps to throttle and hands to the mixer. There is no attitude or rate loop underneath.</li>
 <li><strong>The clips are why the fast eight fails.</strong> At the 3.5 s period the position error saturates, so the policy cannot tell a half-metre lag from a two-metre one. That is the wide orbit in the clip further down.</li>
</ul>
</details>

It is a good controller. It also has habits. On a **5-inch racing quad**, the Flightmare racer from the rl-tools vehicle registry, which sits outside its training range, it hovers a hand's width above the setpoint and overshoots a step climb badly. On the **30 g Crazyflie** it hovers a little low and drops a fraction of its recoveries from a tumbled release. On the **2 kg x500**, PX4's own development quad, it is well behaved until pushed. Nobody injected these. They are what the original RAPTOR weights do, and they are the kind of thing a control engineer would normally go after with a gain table.

<figure class="wm-fig">
 <img src="/images/weight-surgeon/plot_episodes_v3.png" alt="Four panels of height against time. Top row: racer and Crazyflie hover, the original RAPTOR weights settling off the setpoint, the merged controller settling close to it. Bottom row: racer and x500 one metre climbs, the original weights overshooting, the merged controller arriving without overshoot but a little later.">
 <figcaption>One held-out episode per cell, the original RAPTOR weights against the merged controller. The hovers settle nearer the setpoint; the climbs lose their overshoot and gain a slightly slower rise, which is the trade the model chose and wrote down. The Crazyflie's merged-controller hover rings for a second before it settles.</figcaption>
</figure>

## The loop

Every repair runs the same five steps.

<figure class="wm-fig wm-diagram">
 <img src="/images/weight-surgeon/loop_v6.svg" alt="Five boxes left to right: fly, brief, edit, submit, verdict. An amber loop above the edit box is labelled harness: it owns the live weights and the budget, flies each edit and returns the recording. A line underneath says nothing is hidden from the model and nothing is injected.">
 <figcaption>The model works in a sandbox and never touches the live weights directly: every edit and every flight goes through the harness. The verdict is computed on starts it never flew, across every vehicle and scenario, and a run with a regression is a failed run.</figcaption>
</figure>

Nobody told the model how to work, and in every run it did roughly the same thing. It rebuilt the controller and the vehicle dynamics in numpy from the brief's equations and checked its copy against the recording. It formed a physical hypothesis. It scanned weights in its own simulator, sometimes all of them, in the background while it kept reasoning. Then it spent one official rollout to confirm the harness agreed, and submitted with a paragraph naming the mechanism, the alternatives it rejected, and the vehicles the edit might hurt.

## What it found

The racer's overshoot is the cleanest example of the kind of reasoning that came back. The module's input pipeline, which the harness reproduces for every vehicle, clips the position and velocity errors before the network sees them. On a step climb the network stares at a frozen input for most of the way up, keeps ramping collective, and only starts braking when the error unclips. A quadrotor brakes at roughly one g, so the overshoot scales with how hard the vehicle accelerated while blind, and the racer accelerates very hard. The fix was one entry of the GRU input matrix, routing a hidden unit that happens to fire only at the racer's low hover throttle into a gentler vertical gain. One number, and the other vehicles never notice it.

It matters that they never notice, because there is only one set of weights. Nothing here is tuned per platform. An edit meant for the racer has to go through something only the racer switches on, and its hover throttle, far lower than the other two ever fly at, is exactly that.

<figure class="wm-fig wm-video">
 <video autoplay loop muted playsinline controls preload="metadata" aria-label="Two racing quads in a warehouse scene fly a sideways step then a climb. The one with the white beacon climbs past the target and comes back down; the one with the blue beacon stops at the target."><source src="/images/weight-surgeon/isaac_racer_step_v3.mp4" type="video/mp4"></video>
 <figcaption>The same step, flown in Isaac Sim. White beacon: original RAPTOR weights. Blue beacon: merged controller. Same start, same physics, different numbers in the matrix.</figcaption>
</figure>

The fast figure-eight is my favourite, because the account reads like something out of a controls textbook and the fix is something you could actually think if you were tuning a real controller. The 2 kg x500 tracks a leisurely eight fine and then, a little faster, falls off a cliff: every start lands on the same wide lagging orbit. The model's explanation is a resonance. The position and velocity terms are clipped and stop growing with amplitude, the body-rate term is not, so at some speed rate damping becomes the largest thing in the input layer, the motors cannot keep up, and the loop jumps to the lagging branch. The edit scales the roll-rate input column down. 

<figure class="wm-fig">
 <img src="/images/weight-surgeon/plot_cliff_v3.png" alt="Tracking error on a log scale against figure-eight period, faster to the left, for three vehicles with the original RAPTOR weights and the merged controller. The original x500 and racer curves jump by a factor of several at a particular period; the merged curves stay low until a faster period. The Crazyflie crashes at fast periods with either set of weights.">
 <figcaption>Figure-eight tracking error against speed, eight starts per point. The cliff moves to the left for the x500 and the racer. The Crazyflie, at a thrust-to-weight ratio of 1.5, cannot fly the fast eights with either set of weights, and no edit pretended otherwise.</figcaption>
</figure>

## Some more testing before more cool videos

Six ordinary cells first, easiest behaviour first: two hovers, two steps, a figure-eight, a tumbled recovery. Then three scenes where the stock x500 breaks when pushed: the fast eight, a release near inverted, a tall step. Then two merges, because summing the individual edits over-corrects, and **the model had to reconcile its own earlier work**, which it did by recognising three of its racer fixes as the same fix and keeping it once. Every run passed its own cell without a regression elsewhere. The full grid is below; the recoveries are where the change is largest.

<figure class="wm-fig">
 <img src="/images/weight-surgeon/plot_crashes_v3.png" alt="Grouped bars of crash rate for five vehicle and scenario pairs, the original RAPTOR weights against the merged controller. Every merged bar is lower than its original bar; the Crazyflie released near inverted still crashes most of the time.">
 <figcaption>Crash rates on fresh starts, released tilted. Lower everywhere, and still high where the vehicle has no thrust margin to right itself.</figcaption>
</figure>

<figure class="wm-fig">
 <img src="/images/weight-surgeon/final_scorecard.png" alt="A grid of three vehicles by seven scenarios, each cell showing the original metric, the merged metric and the percentage change, shaded blue where the merged controller is better. Almost every cell is blue; one is faintly orange.">
 <figcaption>The merged controller against the original RAPTOR weights on fresh seeds, every vehicle, every scenario. One cell is worse on its headline metric: the x500's 1 m step, by a few percent on RMS error, because it no longer overshoots and rises a touch slower.</figcaption>
</figure>

<figure class="wm-fig wm-video">
 <video autoplay loop muted playsinline controls preload="metadata" aria-label="Two racing quads released nearly upside down in a warehouse. The white-beacon quad falls and freezes red on the floor; the blue-beacon quad rights itself and climbs back to the setpoint."><source src="/images/weight-surgeon/isaac_racer_flip_v2.mp4" type="video/mp4"></video>
 <figcaption>Racer released near inverted, in Isaac Sim. White beacon: original RAPTOR weights, frozen red where the crash rule fired. Blue beacon: merged controller.</figcaption>
</figure>

## Testing with SITL and more realistic simulations

<figure class="wm-fig wm-diagram">
 <img src="/images/weight-surgeon/deploy_sitl_v7.svg" alt="The merged controller fans out into two lanes that end in the same Isaac Sim box. Top: the x500 lane, a PX4 box labelled mc_raptor exchanging actuator commands and IMU, GPS and barometer data with a Pegasus MAVLink box inside Isaac Sim. Bottom: the racer lane, our 100 Hz script with observation, GRU and rotor stages exchanging forces and pose with Isaac Sim directly. Frames go out as mp4.">
 <figcaption>Two routes out of numpy, both landing in Isaac Sim. The racer flies through our own policy and rotor model. The x500 goes through PX4's real module, which loads the same policy.tar the harness exports, with Isaac as the plant behind a MAVLink bridge.</figcaption>
</figure>

For the racer, Isaac Sim provides two rigid bodies with the racer's mass and inertia, a warehouse and a camera. Our numpy policy and our rotor model run on each body at 100 Hz, PhysX integrates, RTX renders. Before trusting any clip I compared the Isaac flights with the numpy simulator on the same starts, and for hover, the step and the figure-eight they agree to about a centimetre. Tumbles diverge after half a second, as chaotic things do.

<figure class="wm-fig wm-video">
 <video autoplay loop muted playsinline controls preload="metadata" aria-label="Two racing quads fly a three-dimensional course through a warehouse. The white-beacon quad cuts corners and lags; the blue-beacon quad holds the line."><source src="/images/weight-surgeon/isaac_racer_course_v3.mp4" type="video/mp4"></video>
 <figcaption>A 3D course neither controller was tuned on. White: original RAPTOR weights. Blue: merged controller.</figcaption>
</figure>

The x500 route is **the one that matters to anyone running PX4**. The harness exports a `policy.tar` the module accepts as its own, self-test block included. The file goes where the module loads it, PX4 SITL boots with the x500's rotor geometry, and the real `mc_raptor` runs the inference in C++ with EKF2 estimating the state from simulated sensors. Isaac Sim is the plant on the other end of a MAVLink link, through the Pegasus Simulator backend, in lockstep. Nothing in PX4 knows the weights were edited. On the fast figure-eight the original weights settle on the same wide lagging orbit they found in numpy, and the merged controller flies the eight.

<figure class="wm-fig wm-video">
 <video autoplay loop muted playsinline controls preload="metadata" aria-label="Two side-by-side views of an x500 quad in a warehouse, flown by PX4. Left, the original RAPTOR weights circle wide of a red figure-eight reference; right, the merged controller traces it closely in blue."><source src="/images/weight-surgeon/px4_x500_eight_fast_side_by_side_px4logo.mp4" type="video/mp4"></video>
 <figcaption>PX4 SITL in Isaac Sim on the 3.5 s figure-eight, both runs aligned on the moment the RAPTOR mode took over. Left: original RAPTOR weights, real mc_raptor inference. Right: merged controller, loaded by the module from our policy.tar. Red is the reference the module generated from its activation pose, blue is the flown path.</figcaption>
</figure>

## Some thoughts

All of this is simulation. The numbers come from the simulator RAPTOR was trained in, with the module's observation pipeline copied from the PX4 source, and the PX4 flights use a Pegasus Iris body carrying the x500's mass and thrust, not the training twin.

One run failed, and it belongs here with the rest. After the merge I asked for one more fix: the x500 thrown into the air, which crashes about one start in seven. Every crash is an upward throw. The controller cuts all four motors to stop the climb and, in the same hundred milliseconds, asks for a braking tilt, and an x500 at zero throttle has no torque left to stop the rotation it just started. It ends upside down. The edit took the horizontal-velocity inputs away from the hidden unit that fires at low throttle, and the crash rate fell from 14.7 % to 5.7 %.

The verdict still said no. A hard upward throw makes the x500 cut its motors, and for that moment it looks like a racer to the network: the low-throttle unit from the step fix switches on. Weakening it saved the x500 and cost the racer, which flies on that unit all the time. Its figure-eight got 12 % worse, past the 10 % a run is allowed. So the merged controller in the videos does not carry the throw fix. A version that only acts when the motors are actually floored is what I would try next.

I would also not read too much into how easily the model found these. The brief hands it the physics, the recording and the weights as text, which is more than I usually get, and 2,084 numbers is few enough to scan one at a time. A controller with a million weights is a different problem, and I have not tried it.

Within those limits the result holds. The verdict was computed on seeds the model never flew, every run that went into the merged controller cleared it, and every edit arrived with a mechanism I could check against the recording. That is more than my gain tables usually come with.

Code, briefs, transcripts and every edit script: soon to be public.
