/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Create a random engine for distributions
  default_random_engine gen;
  
  // Define the number of particles on the filer
  num_particles = 50;
  
  // Gaussian noise across each measurement
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  
  for(int i =0; i < num_particles; i ++){
    // Create a new Particle
    Particle p;
    
   	// Initialize each particle with estimates on the position
    p = {.id=i, .x=dist_x(gen), .y=dist_y(gen), .theta=dist_theta(theta), .weight=1}
    
    // Add the particle to the particles vector
    particles.push_back(p);
  }
  
  // The filter is now initialized
  is_initialized = true
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Create a random engine for distributions
  default_random_engine gen;
  
  // Gaussian distributions around 0
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  
  for (int i = 0; i < num_particles; i ++){
    
    theta_0 = particle[i].theta;
    
    // Determine whether the yaw rate is approx. zero
    if(fabs(yaw_rate) < 1/10000){
      particle[i].x += velocity * delta_t + velocity * cos(theta_0);
      particle[i].y += velocity * delta_t + velocity * sin(theta_0);
    } 
    else {
      particle[i].x += velocity/yaw_rate * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
      particle[i].y += velocity/yaw_rate * (cos(theta_0) - cos(theta_0 + yaw_rate * delta_t));
      particle[i].theta += yaw_rate * delta_t;
    }
    
    // Adding Random Gaussian Noise
    particle[i].x += dist_x(gen);
    particle[i].y += dist_y(gen);
    particle[i].theta += dist_theta(gen);
      
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // For each particle - determine which landmarks are within range
  for (int i = 0; i < num_particles; i ++){
    
    double p_x = particle[i].x;
    double p_y = particle[i].y;
    double p_theta = particle[i].theta;
    
    // Convert observations from VEHICLE's coordinates to MAP's coordinates
    vector<LandmarkObs> m_obs;
    
    for (int j = 0; j < observations.size(); j ++){

      m_obs[j].x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      m_obs[j].y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;

    }
    
    // Find landmarks within the immediate vicinity of the vehicle
    vector<LandMartkObs> predictions;
     
    for (int j = 0; j < map_landmarks.landmark_list.size(); j ++){
      
      int l_id = map_landmarks.landmark_list[j].id;
      float l_x = map_landmarks.landmark_list[j].x;
      float l_y = map_landmarks.landmark_list[j].y;
       
      // Potential Predictions for each particle
      if(fabs(p_x - l_x) <= sensor_range && fabs(p_y - l_y) <= sensor_range){
        LandmarkObs obs_p = {.id=l_id, .x=lm_x, .y=lm_y};
        predictions.push_back(obs_p);
      }    
    }   
    
    // TODO: Link predictions with map observations
    // TODO: Update the weight of the particle
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}