#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

	///* initially set to false, set to true in first call of ProcessMeasurement
	bool is_initialized_;

	///* if this is false, laser measurements will be ignored (except for init)
	bool use_laser_;

	///* if this is false, radar measurements will be ignored (except for init)
	bool use_radar_;

	///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	VectorXd x_;

	///* state covariance matrix
	MatrixXd P_;

	///* predicted sigma points matrix
	MatrixXd Xsig_pred_;

	///* time when the state is true, in microseconds
	long long time_us_;

	///* Process noise standard deviation longitudinal acceleration in m/s^2
	double std_a_;

	///* Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd_;

	///* Laser measurement noise standard deviation position1 in m
	double std_laspx_;

	///* Laser measurement noise standard deviation position2 in m
	double std_laspy_;

	///* Radar measurement noise standard deviation radius in m
	double std_radr_;

	///* Radar measurement noise standard deviation angle in rad
	double std_radphi_;

	///* Radar measurement noise standard deviation radius change in m/s
	double std_radrd_ ;

	///* Weights of sigma points
	VectorXd weights_;

	///* State dimension
	int n_x_;

	///* Augmented state dimension
	int n_aug_;

	///* Number of Sigma Points to generate
	int n_sig_;

	///* Sigma point spreading parameter
	double lambda_;

	MatrixXd R_lidar_;
	MatrixXd R_radar_;
	MatrixXd H_lidar_;
	MatrixXd H_radar_;

	/**
	* Constructor
	*/
	UKF();

	/**
	* Destructor
	*/
	virtual ~UKF();


	/**
	* ProcessMeasurement
	* @param meas_package The latest measurement data of either radar or laser
	*/
	void ProcessMeasurement(MeasurementPackage meas_package);

private:
	// This is the number of States that our System have. This way I can avoid hard coded values and make it more modular and easy to change
	const int StateSize_ = 5;
	/*
	The states are: (The STATE of the system represents the parameters of the object detected by the sensors)
	px: Position coordinate X
	py: Position coordinate Y
	v:  Velocity Magnitude
	phi: Yaw angle of the vehicle wrt X axix: Angle formed by the longitudinal axix across the vehicle and the inertial frame X axis
	phi_dot: Yaw rate or Angular velocyty of the turn. Unkown at the initialization point
	*/

	//create sigma point matrix
	MatrixXd Xsig_aug_;

	///* RADAR NIS
	double NIS_radar_;

	///* LIDAR NIS
	double NIS_lidar_;

	/**
	* Initialize the KF matrices and parameters
	* @param meas_package The latest measurement data of either radar or laser
	* Initialize the KF matrices and parameters
	*/
	void Initialize(MeasurementPackage meas_package);

	/**
	* Prediction Predicts sigma points, the state, and the state covariance
	* matrix
	* @param delta_t Time between k and k+1 in s
	*/
	void Prediction(double delta_t);

	/**
	* Updates the state and the state covariance matrix using a laser measurement
	* @param meas_package The measurement at k+1
	*/
	void UpdateLidar(MeasurementPackage meas_package);

	/**
	* Updates the state and the state covariance matrix using a radar measurement
	* @param meas_package The measurement at k+1
	*/
	void UpdateRadar(MeasurementPackage meas_package);

	void AugmentedSigmaPoints();
	void SigmaPointPrediction(double dt);
	void PredictMeanAndCovariance();

	/**
	*  Angle normalization to [-Pi, Pi]
	*  @param angle
	*/
	void AngNorm(double *ang);

	void UpdateState(MatrixXd Zsig, MatrixXd z_pred, MatrixXd S, int n_z, MeasurementPackage meas_package);
};

#endif /* UKF_H */
