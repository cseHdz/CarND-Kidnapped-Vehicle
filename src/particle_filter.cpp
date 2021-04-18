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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  // Create a random engine for distributions
  default_random_engine gen;
  
  // Define the number of particles on the filer
  num_particles = 100;
  
  // Gaussian noise across each measurement
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for(int i =0; i < num_particles; i ++){
    // Create a new Particle
    Particle p;
    
   	// Initialize each particle with estimates on the position
    p.id = i;
    p.weight = 1.0;
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    
    // Add the particle to the particles vector
    particles.push_back(p);
    weights.push_back(1.0);
  }
  
  // The filter is now initialized
  is_initialized = true;
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
  
  for (int i = 0; i < num_particles; i ++){
    
    double theta_0 = particles[i].theta;
    
    double x_f = particles[i].x;
    double y_f = particles[i].y;
    double theta_f = theta_0;
    
    // Determine whether the yaw rate is approx. zero
    if(abs(yaw_rate) < 0.000001){
      x_f += (velocity * delta_t * cos(theta_0));
      y_f += (velocity * delta_t * sin(theta_0));
    } 
    else {
      theta_f += (yaw_rate * delta_t);
       
      if (theta_f >= 2*M_PI){ theta_f = theta_f - 2*M_PI; }
      if (theta_f < 0.0 ) { theta_f = theta_f + 2*M_PI; }
      
      x_f += (velocity/yaw_rate) * (sin(theta_f) - sin(theta_0));
      y_f += (velocity/yaw_rate) * (cos(theta_0) - cos(theta_f));  
      
      
    }
    
    // Gaussian distributions around 0
    normal_distribution<double> dist_x(x_f, std_pos[0]);
    normal_distribution<double> dist_y(y_f, std_pos[1]);
    normal_distribution<double> dist_theta(theta_f, std_pos[2]);
    
    // Adding Random Gaussian Noise
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (unsigned int i = 0; i < observations.size(); i ++){
    
    // Iterate through each particle to determine how far it is.
    int best_id;
    double min_dist = 999999.0; 
    for (unsigned int j =0; j < predicted.size(); j ++){
     
      double o_dist = dist(observations[i].x, predicted[j].x,
                           observations[i].y, predicted[j].y);
      
      // Determine if the landmark is the closest
      if(o_dist < min_dist){
        min_dist = o_dist;
        best_id = predicted[j].id;
      }
    }
    
    observations[i].id = best_id;
    
  }
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
    
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    double p_weight = 1.0;
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    // Convert observations from VEHICLE's coordinates to MAP's coordinates
    for (unsigned int j = 0; j < observations.size(); j ++){

      double obs_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      double obs_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;

      int best_id;
      double min_dist = numeric_limits<double>::max();
      
      for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k ++){

        int l_id = map_landmarks.landmark_list[k].id_i;
        float l_x = map_landmarks.landmark_list[k].x_f;
        float l_y = map_landmarks.landmark_list[k].y_f;

        // Potential Predictions for each particle
        if(fabs(p_x - l_x) <= sensor_range && fabs(p_y - l_y) <= sensor_range){
          
          double o_dist = dist(obs_x, l_x, obs_y, l_y);
          
          // Determine if the landmark is the closest
          if(o_dist < min_dist){
            min_dist = o_dist;
            best_id = l_id;
          }
        }    
      }
      
      // Get coordinates of best landmark
      float best_x = map_landmarks.landmark_list[best_id].x_f;
      float best_y = map_landmarks.landmark_list[best_id].y_f;     
      
      // Update the weight of the particle
      double gauss_norm;
      double exponent;
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];

      // Multi-gaussian distribution
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      exponent = (pow(obs_x - best_x, 2) / (2 * pow(sig_x, 2))) 
                  + (pow(obs_y - best_y, 2) / (2 * pow(sig_y, 2)));
      
      p_weight *= gauss_norm * exp(-exponent);     
          
      associations.push_back(best_id+1);
      sense_x.push_back(obs_x);
      sense_y.push_back(obs_y);     
    }
    
    SetAssociations(particles[i],associations,sense_x,sense_y);
    
    particles[i].weight = p_weight;
    weights[i] = p_weight; 
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
   // Create a random engine for distributions
  default_random_engine gen;
  
  vector<Particle> r_particles;
  
  discrete_distribution<int> d(weights.begin(), weights.end());
 
  for(int i = 0; i < num_particles; ++i) {

    r_particles.push_back(particles[d(gen)]);
  }  
  particles = std::move(r_particles);
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