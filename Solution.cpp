/*
	Holds modified functions for the main function that will use new classes.
*/
#include "Solution.h"

Solution::Solution() {


}

Solution::~Solution() {

}

void Solution::preprocessing(IloEnv& env, TSLP& prob)
{
	// Rearrange the second-stage constraints data and variable bound data s.t.
	// 1 - eliminate unnecessary constraints like <= IloInfinity
	// 2 - all constraints are <=, >= or = constraints, if both bounds exist, append to the end of the constraint list
	// 3 - all variables are >= are = constraints
	for (int j = 0; j < prob.firstvarlb.getSize(); ++j)
	{
		if (prob.firstvarlb[j] == -1e10)
			prob.firstvarlb[j] = -IloInfinity;
		if (prob.firstvarub[j] == 1e10)
			prob.firstvarub[j] = IloInfinity;
	}

	for (int j = 0; j < prob.secondvarlb.getSize(); ++j)
	{
		if (prob.secondvarlb[j] == -1e10)
			prob.secondvarlb[j] = -IloInfinity;
		if (prob.secondvarub[j] == 1e10)
			prob.secondvarub[j] = IloInfinity;
	}

	// Construct CoefMatXf:
	prob.CoefMatXf = MatrixXf(prob.nbSecRows, prob.nbFirstVars);
	prob.CoefMatXf.setConstant(0);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind < prob.nbFirstVars)
				prob.CoefMatXf(i, ind) = prob.CoefMat[i][j];
		}
	}
}


void Solution::solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf)
{
	// Benders: single cut
	double starttime = clock.getTime();
	bool feas_flag = 1;

	// Construct Master Problem
	MasterProblem master(env, prob, stat, clock, samples, xiterateXf);
	// Add first-stage constraints and objective
	master.define_lp_model();

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subproblem subp(env, prob);

	subp.construct_second_opt(env, prob);
	subp.construct_second_feas(env, prob);

	int nbScens = samples.size();

	while ((stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) > 1e-6 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		IloNumArray xvals(env);
		double thetaval;
		double lasttime = clock.getTime();
		master.getCplex().solve();
		cout << "master status = " << master.getCplex().getStatus() << endl;
		if (master.getCplex().getStatus() == IloAlgorithm::Unbounded)
		{
			IloEnv meanenv;
			stat.relaxobjval = solve_mean_value_model(prob, meanenv, xvals, samples);
			meanenv.end();
		}
		else
		{
			stat.relaxobjval = master.getCplex().getObjValue();
			master.getCplex().getValues(xvals, master.getX());
			thetaval = master.getCplex().getValue(master.getTheta());
		}
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterateXf(j) = xvals[j];
		stat.mastertime += clock.getTime() - lasttime;
		double feasbound = 0.0;
		IloExpr lhsaggr(env);
		double lhsaggrval = 0;
		lhsaggr += master.getTheta();
		lhsaggrval += thetaval;
		double rhsaggr = 0;
		lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each scenario
			IloNumArray duals(env);
			bool feasflag;
			double subobjval = subp.solve(prob, xvals, duals, samples[k], feasflag, subp.calculate_bd(prob, xvals, samples[k]));
			

			VectorXf dualvec(prob.nbSecRows + prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows + prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = 0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggrval += xvals[j] * opt_cut_coef[j] * 1.0 / nbScens;
						lhsaggr += master.getX()[j] * opt_cut_coef[j] * 1.0 / nbScens;
						sum_xvals += opt_cut_coef[j] * xvals[j];
					}
				}
				double rhssub = subobjval + sum_xvals;
				rhsaggr += rhssub * 1.0 / nbScens;
				feasbound += subobjval;
			}
			else
			{
				cout << "infeasible!" << endl;
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
						lhssub += master.getX()[j] * feas_cut_coef[j];
				}
				double rhssub = sum_xvals + subobjval;
				stat.num_feas_cuts++;
				master.getModel().add(lhssub >= rhssub);
				lhssub.end();
			}
		}
		stat.subtime += clock.getTime() - lasttime;
		if (feas_flag == 1)
		{
			feasbound = feasbound * 1.0 / nbScens;
			for (int j = 0; j < prob.nbFirstVars; ++j)
				feasbound += xvals[j] * prob.objcoef[j];
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			master.getModel().add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		else
			cout << "Infeasible! Generating feasibility cuts!" << endl;
		lhsaggr.end();
		xvals.end();
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) << endl;
		stat.solvetime = clock.getTime() - starttime;
		cout << "stat.solvetime = " << stat.solvetime << endl;
		if (stat.solvetime > 10800)
			break;
	}
}



void Solution::solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option)
{
	// Level: level method starts with the mean-value solution
	// option = 0: solving SAA for getting a candidate solution; option = 1: solving SAA for evaluating a given solution using CI
	// first solve a mean-value problem
	double accuracy;
	if (option == 0)
	{
		// For getting a candidate solution
		accuracy = 1e-6;
	}
	if (option == 1)
	{
		// For evaluating a given solution using CI
		accuracy = 1e-4;
	}
	double starttime = clock.getTime();
	// Create and define level and quadratic Master Problems
	IloEnv meanenv;
	IloEnv lenv;
	MasterProblem master(env, prob, stat, clock, samples, xiterateXf, meanenv, lenv);
	master.define_lp_model();
	master.define_qp_model();

	bool feas_flag = 1;

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subproblem subp(env, prob);
	subp.construct_second_opt(env, prob);
	subp.construct_second_feas(env, prob);

	int nbScens = samples.size();

	// This part needs to be changed into Level method iteration
	while ((stat.feasobjval - stat.relaxobjval) * 1.0 / (fabs(stat.feasobjval) + 1e-10) > accuracy || feas_flag == 0)
	{
		feas_flag = 1;
		IloEnv env2;
		stat.iter++;
		IloNumArray xiteratevals(env2, prob.nbFirstVars);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			master.setXiterateXF(j, master.getXiterateVal(j));
			xiteratevals[j] = master.getXiterateVal(j);
		}
		double feasbound = 0.0;
		for (int j = 0; j < prob.nbFirstVars; ++j)
			feasbound += master.getXiterateVal(j) * prob.objcoef[j];

		IloExpr lhsaggr(env);
		lhsaggr += master.getTheta();
		IloExpr llhsaggr(lenv);
		llhsaggr += master.getLtheta();
		double rhsaggr = 0;
		double lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each partition
			IloNumArray duals(env2);
			bool feasflag;
			double subobjval = subp.solve(prob, xiteratevals, duals, samples[k], feasflag, subp.calculate_bd(prob, master.getMeanxvals(), samples[k]));
			VectorXf dualvec(prob.nbSecRows + prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows + prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = opt_cut_coef.dot(xiterateXf);
				double rhssub = subobjval + sum_xvals;
				feasbound += subobjval * 1.0 / nbScens;
				rhsaggr += rhssub * 1.0 / nbScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggr += master.getX()[j] * opt_cut_coef[j] * 1.0 / nbScens;
						llhsaggr += master.getLx()[j] * opt_cut_coef[j] * 1.0 / nbScens;
					}
				}

			}
			else
			{
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose() * dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				IloExpr llhssub(lenv);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
					{
						llhssub += master.getLx()[j] * feas_cut_coef[j];
						lhssub += master.getX()[j] * feas_cut_coef[j];
					}
				}
				double rhssub = sum_xvals + subobjval;
				stat.num_feas_cuts++;
				master.getModel().add(lhssub >= rhssub);
				master.getLevelmodel().add(llhssub >= rhssub);
				lhssub.end();
				llhssub.end();
			}
		}
		xiteratevals.end();
		stat.subtime += clock.getTime() - lasttime;
		if (feas_flag == 1)
		{
			//cout << "feasbound = " << feasbound << ", feasobjval = " << stat.feasobjval << endl;
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			master.getLevelmodel().add(llhsaggr >= rhsaggr);
			master.getModel().add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		lhsaggr.end();
		llhsaggr.end();
		// Now solve the master, get a lower bound
		lasttime = clock.getTime();
		master.getCplex().solve();
		stat.relaxobjval = master.getCplex().getObjValue();
		stat.mastertime += clock.getTime() - lasttime;

		// Now solve the qp level problem
		// update the upper bound, (1-\lambda)F^{k+1}+\lambda F^*

		master.getRangeub().setUB(0.5 * stat.relaxobjval + 0.5 * stat.feasobjval);
		IloExpr objExpr(lenv);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			objExpr += master.getLx()[j] * master.getLx()[j];
			objExpr -= master.getLx()[j] * 2 * master.getXiterateVal(j);
		}
		master.getLobj().setExpr(objExpr);
		objExpr.end();

		double startqptime = clock.getTime();
		master.getLevelcplex().solve();
		stat.qptime += clock.getTime() - startqptime;
		IloNumArray lxval(lenv);
		master.getLevelcplex().getValues(lxval, master.getLx());
		for (int j = 0; j < prob.nbFirstVars; ++j)
			master.setXiterateVal(j, lxval[j]);
		lxval.end();
		env2.end();
		//cout << "relaxobjval = " << stat.relaxobjval << endl;
		//cout << "feasobjval = " << stat.feasobjval << endl;
		//cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime() - starttime;
		if (stat.solvetime > 10800)
			break;
	}

}



bool Solution::solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime)
{
	// option = 0: Just use the relative opt gap, solution mode: 1e-6; option = 1: Use the adaptive sampling type threshold: when opt gap is small relative to the sample error, option = 2: use relative opt gap, evaluation mode: 1e-4
	// initial = 0: a stability center is provided as xiterteXf from the previous iteration; initial = 1: the very first sampled problem solved
	// Unified coarse/fine oracle as a partly inexact oracle


	bool returnflag = 1;
	if (option == 1)
		cout << "begin solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "begin evaluation mode..." << endl;
	IloEnv env2;
	IloNumArray xvals(env2);
	double eval_threshold = 1e-4;
	double starttime = clock.getTime();
	double fmean;
	bool feas_flag = 0;
	IloNumArray stab_center(env, prob.nbFirstVars);
	vector< vector<double> > cutcoefs; // We may do some cut screening later
	vector<double> cutrhs;
	IloRangeArray cuts(env);
	double warmtemp = clock.getTime();

	// Create Partition object
	IloObjective QPobj;
	IloEnv meanenv;
	MasterProblem masterP(env, prob, stat, clock, samples, xiterateXf);
	Subproblem subP(env, prob);
	Partition part_call(masterP, subP);

	if (initial == 0)
	{
		// One of the later iterations
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = xiterateXf(j);
		// We will also do some warm starts here using recorded dual information from dualCollection
		fmean = part_call.solve_warmstart(meanenv, prob, samples, stab_center, dualInfoCollection, cutcoefs, cutrhs, rhsvecs, xvals, clock, stat);

		// Add cuts
		for (int l = 0; l < cutcoefs.size(); ++l)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				lhs += part_call.masterProb.getX()[j] * (prob.objcoef[j] - cutcoefs[l][j] * 1.0 / samples.size());
			// Just tempararily set an UB, will be updated in the inner loop any way
			IloRange range(env, -IloInfinity, lhs, 0);
			cuts.add(range);
			lhs.end();
		}
		// This is only temporary
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		part_call.partition.push_back(all);
	}
	else
	{
		// Solve the mean-value problem first
		fmean = solve_mean_value_model(prob, meanenv, xvals, samples);
		// Set stab_center to be 0 initially
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = 0;
		// Initial partition: everybody is together, i.e., the mean-value problem
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		part_call.partition.push_back(all);
	}
	stat.warmstarttime = clock.getTime() - warmtemp;
	meanenv.end();
	IloRangeArray center_cons(env);
	setup_bundle_QP(env, prob, part_call.masterProb.getCplex(), part_call.masterProb.getModel(), part_call.masterProb.getX(), stab_center, QPobj, cuts, center_cons);
	//***** part_call.masterProb.setup_bundle_QP(stab_center, QPobj, cuts, center_cons);
	bool firstloop = 1;
	double descent_target;
	double opt_gap;
	double samplingError = 0; // samplingError will be updated whenever there is a chance to loop through all scenarios
	bool loopflag = 1;
	while (loopflag == 1 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		//cout << "stat.iter = " << stat.iter << endl;
		stat.partitionsize += part_call.partition.size();
		double feasboundscen = 0.0;
		VectorXf cutcoefscen(prob.nbFirstVars);
		cutcoefscen.setZero();
		VectorXf aggrCoarseCut(prob.nbFirstVars);
		vector<VectorXf> partcoef(part_call.partition.size());
		vector<double> partrhs(part_call.partition.size());
		double coarseLB, coarseCutRhs;
		vector<double> scenObjs; // record scenario objectives
		if (firstloop == 1)
			stat.relaxobjval = fmean;
		else
		{
			double lasttime = clock.getTime();
			coarseLB = part_call.coarse_oracle(env, prob, xvals, feasboundscen, cutcoefscen, stat, center_cons, stab_center, cuts, cutrhs, aggrCoarseCut, coarseCutRhs, partcoef, partrhs, starttime, clock, scenObjs, samples, dualInfoCollection, rhsvecs, option);

			stat.solvetime = clock.getTime() - starttime;
			if (stat.solvetime > remaintime)
			{
				returnflag = 0;
				break;
			}
			descent_target = prob.kappaf * stat.relaxobjval + (1 - prob.kappaf) * stat.feasobjval;
			// Here opt_gap is updated since stat.relaxobjval may have been updated during coarse_oracle()
			opt_gap = stat.feasobjval - stat.relaxobjval;
			if (option == 1 && samplingError > opt_gap * 10)
			{
				loopflag = 0;
				continue;
			}
			if (option == 0 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= prob.eps)
			{
				loopflag = 0;
				continue;
			}
			if (option == 2 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= eval_threshold)
			{
				//cout << "opt_gap = " << opt_gap << ", feasobjval = " << stat.feasobjval << endl;
				loopflag = 0;
				continue;
			}
			if (coarseLB > descent_target)
			{
				// Cannot achieve descent target, so just use coarse oracle
				IloExpr lhs(env);
				vector<double> tempcut(prob.nbFirstVars);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(aggrCoarseCut[j]) > 1e-7)
					{
						lhs += part_call.masterProb.getX()[j] * (prob.objcoef[j] - aggrCoarseCut[j] * 1.0 / samples.size());
						tempcut[j] = aggrCoarseCut[j];
					}
					else
					{
						lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						tempcut[j] = 0;
					}
				}
				// Just tempararily set an UB, will be updated in the inner loop any way
				IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
				part_call.masterProb.getModel().add(range);
				cuts.add(range);
				cutcoefs.push_back(tempcut);
				cutrhs.push_back(coarseCutRhs);
				lhs.end();
				stat.num_opt_cuts++;
				if (loopflag != 0)
					continue;
			}
		}
		// Begin partition refinement
		vector<Component> new_partition;
		bool fullupdateflag = 0;
		if (firstloop == 1)
		{
			// just solve all the scenario subproblems
			feas_flag = part_call.refine_full(env, env2, prob, xvals, feasboundscen, cutcoefscen, new_partition, stat, clock, scenObjs, dualInfoCollection, rhsvecs, option);
			fullupdateflag = 1;
		}
		else
		{
			// Solve scenario subproblems component by component, stop when hopeless to achieve descent target
			feas_flag = part_call.refine_part(env, env2, prob, xvals, new_partition, stat, clock, partcoef, partrhs, descent_target, fullupdateflag, coarseLB, aggrCoarseCut, scenObjs, samples, dualInfoCollection, rhsvecs, option);
		}
		if (feas_flag == 1 && fullupdateflag == 1)
		{
			if (scenObjs.size() != samples.size())
			{
				cout << "scenObjs.size != samples.size" << endl;
				exit(0);
			}
			// A new candidate is available, update UB
			if (firstloop == 1)
			{
				double feasbound = 0.0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
					feasbound += xvals[j] * prob.objcoef[j];
				feasbound += feasboundscen * 1.0 / samples.size();
				if (feasbound < stat.feasobjval - 1e-5)
					stat.feasobjval = feasbound;
				// update stablization center
				for (int j = 0; j < prob.nbFirstVars; ++j)
					stab_center[j] = xvals[j];
				computeSamplingError(samplingError, scenObjs);
			}
			else
			{
				if (coarseLB < stat.feasobjval - 1e-5)
					stat.feasobjval = coarseLB;
				// update stablization center
				if (coarseLB < descent_target)
				{
					computeSamplingError(samplingError, scenObjs);
					for (int j = 0; j < prob.nbFirstVars; ++j)
						stab_center[j] = xvals[j];
					// Begin bundle compression
					if (cuts.getSize() >= 100)
					{
						// Only do bundle compression if maintaining too many cuts in the bundle
						double maxval = -1e5;
						vector<double> cutvals(cuts.getSize());
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							double cutval = 0;
							for (int jj = 0; jj < prob.nbFirstVars; ++jj)
							{
								if (fabs(cutcoefs[j][jj]) > 1e-7 && fabs(stab_center[jj]) > 1e-7)
									cutval -= cutcoefs[j][jj] * stab_center[jj];
							}
							cutval = cutval * 1.0 / samples.size();
							cutval += cutrhs[j];
							cutvals[j] = cutval;
							if (cutval > maxval)
								maxval = cutval;
						}
						vector<bool> cutactive(cuts.getSize(), 0);
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (maxval - cutvals[j] < 1e-5)
								cutactive[j] = 1;
							else
							{
								// remove the cut
								part_call.masterProb.getModel().remove(cuts[j]);
								cuts[j].end();
							}
						}
						IloRangeArray newcuts(env);
						vector< vector<double> > newcutcoefs;
						vector<double> newcutrhs;
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (cutactive[j] == 1)
							{
								newcuts.add(cuts[j]);
								newcutcoefs.push_back(cutcoefs[j]);
								newcutrhs.push_back(cutrhs[j]);
							}
						}
						cutcoefs = newcutcoefs;
						cuts = newcuts;
						cutrhs = newcutrhs;
					}
				}
			}
		}
		opt_gap = stat.feasobjval - stat.relaxobjval;
		//cout << "stat.iter = "<< stat.iter << ", relaxobjval = " << stat.relaxobjval << ", feasobjval = " << stat.feasobjval << ", optimality gap = " << opt_gap << ", samplingError = " << samplingError << endl;
		if (option == 0 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= 1e-6)
			loopflag = 0;
		if (option == 1 && samplingError > opt_gap * 10)
			loopflag = 0;
		if (option == 2 && opt_gap * 1.0 / (fabs(stat.feasobjval) + 1e-10) <= eval_threshold)
			loopflag = 0;
		//if (partition.size() == new_partition.size() && feas_flag == 1)
		//	loopflag = 0;
		//cout << "After updates, loopflag = " << loopflag << ", feas_flag = " << feas_flag << ", partition.size = " << partition.size() << ", new_partition.size = " << new_partition.size() << endl;
		if (loopflag == 1 && firstloop == 0)
		{
			part_call.partition = new_partition;
			// Need to add an aggregated cut, if feasible: maybe a fine oracle, or a mixed fine/coarse oracle
			if (feas_flag == 1)
			{
				if (firstloop == 1)
				{
					double lhsval = 0;
					vector<double> tempcut(prob.nbFirstVars);
					IloExpr lhs(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(cutcoefscen[j]) > 1e-7)
						{
							lhs += part_call.masterProb.getX()[j] * (-cutcoefscen[j] * 1.0 / samples.size() + prob.objcoef[j]);
							lhsval += xvals[j] * cutcoefscen[j];
							tempcut[j] = cutcoefscen[j];
						}
						else
						{
							tempcut[j] = 0;
							if (fabs(prob.objcoef[j]) > 1e-7)
								lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						}
					}
					double fineCutRhs = (feasboundscen + lhsval) * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -fineCutRhs);
					part_call.masterProb.getModel().add(range);
					cuts.add(range);
					cutcoefs.push_back(tempcut);
					cutrhs.push_back(fineCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
				else
				{
					coarseCutRhs = coarseLB;
					for (int j = 0; j < prob.nbFirstVars; ++j)
						coarseCutRhs -= prob.objcoef[j] * xvals[j];
					IloExpr lhs(env);
					vector<double> tempcut(prob.nbFirstVars);
					double tempval = 0;
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(aggrCoarseCut[j]) > 1e-7)
						{
							lhs += part_call.masterProb.getX()[j] * (-aggrCoarseCut[j] * 1.0 / samples.size() + prob.objcoef[j]);
							tempval += xvals[j] * aggrCoarseCut[j];
							tempcut[j] = aggrCoarseCut[j];
						}
						else
						{
							tempcut[j] = 0;
							lhs += part_call.masterProb.getX()[j] * prob.objcoef[j];
						}
					}
					cutcoefs.push_back(tempcut);
					coarseCutRhs += tempval * 1.0 / samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
					part_call.masterProb.getModel().add(range);
					cuts.add(range);
					cutrhs.push_back(coarseCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
			}
		}
		firstloop = 0;
		stat.solvetime = clock.getTime() - starttime;
		if (stat.solvetime > remaintime)
		{
			returnflag = 0;
			break;
		}
	}
	if (stat.iter == 1) //nearly optimal after the very first iteration
	{
		if (nearOptimal == 0)
			nearOptimal = 1;
		else
			nearOptimal++;
	}
	else
	{
		if (nearOptimal < 3)
			nearOptimal = 0;
	}
	for (int jj = 0; jj < prob.nbFirstVars; ++jj)
		xiterateXf(jj) = stab_center[jj];
	stat.solvetime = clock.getTime() - starttime;
	stab_center.end();
	xvals.end();
	env2.end();
	cuts.end();
	center_cons.end();
	stat.finalpartitionsize = part_call.partition.size();
	if (option == 1)
		cout << "end solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "end evaluation mode..." << endl;
	return returnflag;
}


double Solution::solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples)
{
	IloModel meanmodel(meanenv);
	IloNumVarArray meanx(meanenv, prob.firstvarlb, prob.firstvarub);
	IloNumVarArray meany(meanenv, prob.nbSecVars, -IloInfinity, IloInfinity);
	// first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += meanx[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
		IloRange range(meanenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		meanmodel.add(range);
		lhs.end();
	}
	int nbScens = samples.size();
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += meany[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
			else
				lhs += meanx[ind] * prob.CoefMat[i][j];
		}
		// set constraint bounds
		double bd = 0;
		for (int k = 0; k < nbScens; ++k)
			bd += prob.secondconstrbd[samples[k] * prob.nbSecRows + i];
		bd = bd * 1.0 / nbScens;
		IloRange range;
		if (prob.secondconstrsense[i] == -1)
			range = IloRange(meanenv, bd, lhs, IloInfinity);
		if (prob.secondconstrsense[i] == 0)
			range = IloRange(meanenv, bd, lhs, bd);
		if (prob.secondconstrsense[i] == 1)
			range = IloRange(meanenv, -IloInfinity, lhs, bd);
		meanmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(meanenv);
		lhs += meany[i];
		IloRange range(meanenv, -IloInfinity, lhs, IloInfinity);
		if (prob.secondvarlb[i] != -IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarlb[samples[k] * prob.nbSecVars + i];
			bd = bd * 1.0 / nbScens;
			range.setLB(bd);
		}
		if (prob.secondvarub[i] != IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarub[samples[k] * prob.nbSecVars + i];
			bd = bd * 1.0 / nbScens;
			range.setUB(bd);
		}
		meanmodel.add(range);
		lhs.end();
	}
	IloExpr meanobj(meanenv);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		meanobj += meanx[i] * prob.objcoef[i];

	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			meanobj += meany[i - prob.nbFirstVars] * prob.objcoef[i];
	}
	meanmodel.add(IloMinimize(meanenv, meanobj));
	meanobj.end();
	IloCplex meancplex(meanmodel);
	meancplex.setParam(IloCplex::TiLim, 3600);
	meancplex.setParam(IloCplex::Threads, 1);
	meancplex.setParam(IloCplex::SimDisplay, 0);
	meancplex.setOut(meanenv.getNullStream());
	meancplex.solve();
	meancplex.getValues(meanxvals, meanx);
	double returnval = meancplex.getObjValue();
	meancplex.end();
	meanmodel.end();
	meanx.end();
	meany.end();
	return returnval;
}



void Solution::computeSamplingError(double& samplingError, const vector<double>& scenObjs)
{
	samplingError = 0;
	double sampleMean = 0;
	for (int k = 0; k < scenObjs.size(); ++k)
		sampleMean += scenObjs[k];
	sampleMean = sampleMean * 1.0 / scenObjs.size();
	for (int k = 0; k < scenObjs.size(); ++k)
		samplingError += pow(scenObjs[k] - sampleMean, 2);
	samplingError = sqrt(samplingError) * 1.0 / scenObjs.size();
	// \delta = 1e-3
	if (samplingError < 1e-3 * 1.0 / sqrt(scenObjs.size()))
		samplingError = 1e-3 * 1.0 / sqrt(scenObjs.size());
}



void Solution::setup_bundle_QP(IloEnv& env, const TSLP& prob, IloCplex& cplex, IloModel& model, IloNumVarArray& x, const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons)
{
	model = IloModel(env);
	// first stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			// Perform refinement
			lhs += x[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
	}
	// Adding objective
	QPobj = IloMinimize(env);
	model.add(QPobj);
	IloExpr objExpr(env);
	IloNumVarArray y(env, prob.nbFirstVars, 0, IloInfinity);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		objExpr += y[j];
	QPobj.setExpr(objExpr);
	objExpr.end();

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j] - x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j] + x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	// Add cuts
	for (int l = 0; l < cuts.getSize(); ++l)
		model.add(cuts[l]);
	cplex = IloCplex(model);
	cplex.setParam(IloCplex::TiLim, 3600);
	cplex.setParam(IloCplex::Threads, 1);
	cplex.setParam(IloCplex::BarDisplay, 0);
	cplex.setParam(IloCplex::SimDisplay, 0);
	cplex.setOut(env.getNullStream());
	//cplex.setParam(IloCplex::EpOpt, 1e-4);
}