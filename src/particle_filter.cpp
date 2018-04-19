/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Modified on: Apr 18, 2018
 *      Author: Brian Lee
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

//declare globally static default engine
static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 101;


  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
   *   standard deviation of yaw [rad]]
   */


  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //set the all-particles and all-weights vector size to the num_particles defined
  particles.resize(num_particles);
  weights.resize(num_particles);

  for(int i = 0; i < num_particles; i++){
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
    //TODO: the following initializations need to be looked at again
    //particles[i].associations = {0,0};
    //particles[i].sense_x = {0.0};
    //particles[i].sense_y = {0.0};
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  for (int i = 0; i < num_particles; i++){
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
      else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
      // Add random Gaussian noise
      particles[i].x += dist_x(gen);
      particles[i].y += dist_y(gen);
      particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> candidates, std::vector<LandmarkObs>& trans_observations) {
  //for each map-coordinates translated observation (each element of the vector trans_observations), compute the euclidean distance
  // to each of the candidates (true map landmark locations read in from map_data.txt that are within the sensor's range)
  // and decide which of the translated observations is the closest to each of the candidates, and assign the candidate's landmark id
  // to corresponding translated observations
  int match_id = 0;

  for(int i = 0; i < trans_observations.size(); i++){
    double prev_dist = numeric_limits<double>::max();
    for(int j = 0; j < candidates.size(); j++){
      double new_dist = dist(candidates[j].x, candidates[j].y, trans_observations[i].x, trans_observations[i].y);
      if(new_dist < prev_dist){
        match_id = candidates[j].id;
        prev_dist = new_dist;
      }
    }
    trans_observations[i].id = match_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  //for each particle..
  for(int i = 0; i < num_particles; i++)
  {
    std::vector<LandmarkObs> candidates;

    //get each particle's x, y, theta for the computations that follow
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {

      //get the id and x, y so that data structure conversion can be easily made while being push_back'ing each transformed values to candidates
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;

      //filter out the landmarks within sensor_range based on each particle's x and y position
      if(dist(p_x, p_y, lm_x, lm_y) <= sensor_range)
      {
        candidates.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    //transform the observations coordinates (from particle's vehicle coordinate to the map's coordinate)
    std::vector<LandmarkObs> trans_observations;
    for (int k = 0; k < observations.size(); k++)
    {
      double t_x = cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y + p_x;
      double t_y = sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y + p_y;
      trans_observations.push_back(LandmarkObs{observations[k].id, t_x, t_y});
    }

    //run data association
    dataAssociation(candidates, trans_observations);

    //initialize a weight variable that can be iteratively multiplied for this particle's final weight
    double w = 1.0;

    for (unsigned int j = 0; j < trans_observations.size(); j++) {

      // placeholders for observation and associated prediction coordinates
      double o_x, o_y, pr_x, pr_y;
      o_x = trans_observations[j].x;
      o_y = trans_observations[j].y;

      int associated_prediction = trans_observations[j].id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < candidates.size(); k++) {
        if (candidates[k].id == associated_prediction) {
          pr_x = candidates[k].x;
          pr_y = candidates[k].y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );

      // product of this obersvation weight with total observations weight
      w *= obs_w;
    }

    //after the final weight (w) is computed, store this into this particle's weight variable (particles[i].weight)
    particles[i].weight = w;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  //create re-sampled particles
  std::vector<Particle> rs_particles;

  //update weights with each particle's weight
  for(int i = 0; i < num_particles; i++)
  {
    weights[i] = particles[i].weight;
  }

  //generate random index. indexing starts from 0 so the max value should be num_particles minus 1
  uniform_int_distribution<int> r_dist(0, num_particles-1);
  auto index = r_dist(gen);

  //declare and initialize beta
  double beta = 0.0;

  //get max weight
  double mw = *max_element(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++)
  {
    uniform_int_distribution<int> beta_dist(0, 1);
    auto beta_r = beta_dist(gen);

    beta += beta_r * 2.0 * mw;
    while(beta > weights[index])
    {
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    rs_particles.push_back(particles[index]);
  }

  particles = rs_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
