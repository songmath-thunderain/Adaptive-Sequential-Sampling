#pragma once
#ifndef PARTITION_H
#define PARTITION_H

#include "adaptiveSamples.h"
#include "Subproblem.h"
#include "MasterProblem.h"

using namespace std;
using Eigen::VectorXf;


class Partition {

private:
	vector<Component> partition;

	//masterProblem needs to have an default constructor
	MasterProblem masterProb;

	Subproblem subProb;

	//Coarse Oracle's Helper Functions
	void setAggregatedBounds(const TSLP& prob, IloNumArray& secvarlb, IloNumArray& secvarub, IloNumArray& secconstrbd);

	bool addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection);

	void add_feas_cuts(IloEnv& env, TSLP& prob, IloModel& model, const IloNumVarArray& x, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, int i);

	//Solve Scen Subprobs' Helper Functions
	void simple_refine(const Component& component, const TSLP& prob, const vector<IloNumArray>& extreme_points, const vector<int>& extreme_points_ind, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, vector<Component>& new_partition, vector< vector<int> >& extreme_ray_map);

	bool compare_arrays(const TSLP& prob, const IloNumArray& array1, const IloNumArray& array2);

	void gen_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, const vector<int>& extreme_ray_map, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, const double sum_of_infeas, IloModel& model, const IloNumVarArray& x);


	
	// Refinement helper Functions
	bool calculate_refinement(vector<IloNumArray>& extreme_points, vector<IloNumArray>& extreme_rays, vector<int>& extreme_points_ind, vector<int>& extreme_rays_ind, double& feasboundscen, VectorXf& cutcoefscen, vector<double>& scenObjs, int option, vector<DualInfo>& dualInfoCollection, const IloNumArray& xvals, const vector<VectorXf>& rhsvecs, double& sum_of_infeas/*,Subproblem*/);

	void add_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, STAT& stat);
	

public:
	//Constructor
	Partition(vector<Component>& partition, MasterProblem masterProb, Subproblem subProb);

	//Deconstructor
	~Partition();

	vector<Component>& getPartition();

	double solve_warmstart(IloEnv& env, const TSLP& prob, const vector<int>& samples, const IloNumArray& stab_center, const vector<DualInfo>& dualInfoCollection, vector< vector<double> >& cutcoefs, vector<double>& cutrhs, vector<Component>& partition, const vector<VectorXf>& rhsvecs, IloNumArray& xvals, IloTimer& clock, STAT& stat);

	double coarse_oracle(IloEnv& env, TSLP& prob, IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloCplex& cplex, IloModel& model, const IloNumVarArray& x, STAT& stat, IloRangeArray& center_cons, const IloNumArray& stab_center, IloRangeArray& cuts, const vector<double>& cutrhs, VectorXf& aggrCoarseCut, double& coarseCutRhs, vector<VectorXf>& partcoef, vector<double>& partrhs, double starttime, IloTimer& timer, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

	bool refine_full(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<double>& scenObjs, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);
	// formerly solve_scen_subprobs and solve_scen_subprobs_target
	bool refine_part(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<VectorXf>& partcoef, vector<double>& partrhs, double descent_target, bool& fullupdateflag, double& coarseLB, VectorXf& aggrCoarseCut, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

	//This function should be called from the main file
	void computeSamplingError(double& samplingError, const vector<double>& scenObjs);

};

#endif