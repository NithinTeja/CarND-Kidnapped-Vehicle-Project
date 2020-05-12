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

static std::default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/**
	 * TODO: Set the number of particles. Initialize all particles to
	 *   first position (based on estimates of x, y, theta and their uncertainties
	 *   from GPS) and all weights to 1.
	 * TODO: Add random Gaussian noise to each particle.
	 * NOTE: Consult particle_filter.h for more information about this method
	 *   (and others in this file).
	 */
	if (is_initialized) {
		return;
	}
	num_particles = 20;  // todo: set the number of particles
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	double std_x = std_pos[0];
	double std_y= std_pos[1];
	double std_theta = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);
	if (is_initialized) {
		for (int i = 0; i < num_particles; i++) {
			if (fabs(yaw_rate) < 0.00001) {
				particles[i].x += velocity * delta_t * cos(particles[i].theta);
				particles[i].y += velocity * delta_t * sin(particles[i].theta);
			}
			else {
				particles[i].x += (velocity / yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
				particles[i].y += (velocity / yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
				particles[i].theta += yaw_rate * delta_t;
			}

			particles[i].x += dist_x(gen);
			particles[i].y += dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
		
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
	for (int i = 0; i < observations.size(); ++i) {
		LandmarkObs obs = observations[i];
		double obs_x = obs.x;
		double obs_y = obs.y;
		
		int map_id = -1;
		double min_dist = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); ++j) {
			LandmarkObs pred = predicted[j];
			double pred_x = pred.x;
			double pred_y = pred.y;
			int pred_id = pred.id;

			double curr_dist = dist(obs.x, obs.y, pred.x, pred.y);
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				map_id = pred_id;
			}
		}
		observations[i].id = map_id;
	}
	

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
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
	double sum_w = 0;
	for (int i = 0; i < num_particles; ++i) {
		//pick current particle
		Particle particle = particles[i];

		double std_lm_x = std_landmark[0];
		double std_lm_y = std_landmark[1];
		// vector of landmarks within sensor range
		vector<LandmarkObs> pred;
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			// Get map landmarks x,y and id
			int map_lm_id = map_landmarks.landmark_list[j].id_i;
			float map_lm_x = map_landmarks.landmark_list[j].x_f;
			float map_lm_y = map_landmarks.landmark_list[j].y_f;
			double range = dist(particle.x, particle.y, map_lm_x, map_lm_y);
			if (range <= sensor_range) {
				pred.push_back(LandmarkObs{ map_lm_id,map_lm_x,map_lm_y });
			}
		}

		//Transform observations from Vehicle's coordinate system to Map's coordinate system
		vector<LandmarkObs> t_observations;
		for (int k = 0; k < observations.size(); ++k) {
			LandmarkObs t_obs;
			double obs_x = observations[k].x;
			double obs_y = observations[k].y;
			t_obs.x = particle.x + (cos(particle.theta) * obs_x) - (sin(particle.theta) * obs_y);
			t_obs.y = particle.y + (sin(particle.theta) * obs_x) + (cos(particle.theta) * obs_y);
			t_obs.id = observations[k].id;
			t_observations.push_back(t_obs);
		}
		
		
		// Data Association step - Associate each transformed observation to nearest predicted landmark
		dataAssociation(pred, t_observations);
		//// Create associations, sense_x, and sense_y vectors using observations converted to map coordinates.
		//vector<int> associations;
		//vector<double> sense_x;
		//vector<double> sense_y;
		//for (int k = 0; k < t_observations.size(); ++k) {
		//	associations.push_back(t_observations[k].id);
		//	sense_x.push_back(t_observations[k].x);
		//	sense_y.push_back(t_observations[k].y);
		//}

		//// Use SetAssociations() and above vectors to update current particle.
		//SetAssociations(particles[i], associations, sense_x, sense_y);
		// reinit weight
		particle.weight = 1.0;
		//Update particle weights using Multivariate-Gaussian probability distribution
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs t_obs = t_observations[j];
			//LandmarkObs nearest_map_lm;
			int nearest_map_lm_id = t_obs.id;
			double nearest_map_lm_x =0, nearest_map_lm_y=0;

			for (int k = 0; k < pred.size(); ++k) {
				if (pred[k].id == nearest_map_lm_id) {
					nearest_map_lm_x = pred[k].x;
					nearest_map_lm_y = pred[k].y;
				}
			}
			 //calculate normalization term
			double gauss_norm;
			gauss_norm = 1 / (2 * M_PI * std_lm_x * std_lm_y);

			// calculate exponent
			double exponent;
			exponent = (pow(t_obs.x - nearest_map_lm_x, 2) / (2 * pow(std_lm_x, 2)))
				+ (pow(t_obs.y - nearest_map_lm_y, 2) / (2 * pow(std_lm_y, 2)));

			particle.weight *= gauss_norm* exp(-exponent);
		}
		sum_w += particle.weight;
		particles[i].weight = particle.weight;
		
	}
	std::cout << "sum_w " << sum_w << std::endl;
	 //Normalize weights between 0-1
	//if (sum_w >0) {
	//	for (int i = 0; i < num_particles; ++i) {
	//		particles[i].weight /= sum_w;
	//	}
	//}
	
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	vector<Particle> resample_particles;
	vector<double> weights;
	for (int i = 0; i < num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}
	std::uniform_int_distribution<int> index_dist(0, num_particles-1);
	double max_w = *max_element(weights.begin(), weights.end());
	std::cout << "max_w" << max_w << std::endl;
	std::uniform_real_distribution<double> weight_dist(0, max_w);
	int index = index_dist(gen);
	double beta = 0;
	for (int i = 0; i < num_particles; ++i) {
		beta += 2*weight_dist(gen);
		while (weights[index] < beta) {
			beta = beta - weights[index];
			index = (index + 1) % num_particles;
		}
		resample_particles.push_back(particles[index]);
	}
	particles = resample_particles;
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