/*
   Functions used to execute option 0 in the main function.
   *****Insert description of option 0.*****
*/

/*
  insert function header
*/
void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf)
{
	// Benders: single cut
	double starttime = clock.getTime();
	bool feas_flag = 1;
	IloModel mastermodel(env);
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	// For now we assume an LB for the second-stage obj: 0
	IloNumVar theta(env, 0, IloInfinity);
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		mastermodel.add(range);
		lhs.end();
	}
	// Adding objective
	IloExpr obj(env);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		obj += x[i]*prob.objcoef[i];
	// single cut
	obj += theta;
	mastermodel.add(IloMinimize(env, obj));
	obj.end();
	IloCplex mastercplex(mastermodel);
	mastercplex.setParam(IloCplex::TiLim,10800);
	mastercplex.setParam(IloCplex::Threads, 1);
	// Barrier
	mastercplex.setParam(IloCplex::RootAlg, 2);
	mastercplex.setParam(IloCplex::BarDisplay, 0);
	mastercplex.setParam(IloCplex::SimDisplay, 0);
	mastercplex.setOut(env.getNullStream());
	//mastercplex.setParam(IloCplex::EpOpt, 1e-4);

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	int nbScens = samples.size();

	while ((stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > 1e-6 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		IloNumArray xvals(env);
		double thetaval;
		double lasttime = clock.getTime();
		mastercplex.solve();
		cout << "master status = " << mastercplex.getStatus() << endl;
		if (mastercplex.getStatus() == IloAlgorithm::Unbounded)
		{
			IloEnv meanenv;
			stat.relaxobjval = solve_mean_value_model(prob, meanenv, xvals, samples);
			meanenv.end();
		}
		else
		{
			stat.relaxobjval = mastercplex.getObjValue();
			mastercplex.getValues(xvals, x);
			thetaval = mastercplex.getValue(theta);
		}
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterateXf(j) = xvals[j];
		stat.mastertime += clock.getTime()-lasttime;
		double feasbound = 0.0;
		IloExpr lhsaggr(env);
		double lhsaggrval = 0;
		lhsaggr += theta;
		lhsaggrval += thetaval;
		double rhsaggr = 0;
		lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each scenario
			IloNumArray duals(env);
			bool feasflag;
			double subobjval = subprob(subp, prob, xvals, duals, samples[k], feasflag);
			VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
				double sum_xvals = 0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggrval += xvals[j]*opt_cut_coef[j]*1.0/nbScens;
						lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
						sum_xvals += opt_cut_coef[j]*xvals[j];
					}
				}
				double rhssub = subobjval + sum_xvals;
				rhsaggr += rhssub*1.0/nbScens;
				feasbound += subobjval;
			}
			else
			{
				cout << "infeasible!" << endl;
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
						lhssub += x[j]*feas_cut_coef[j];
				}
				double rhssub = sum_xvals+subobjval;
				stat.num_feas_cuts++;
				mastermodel.add(lhssub >= rhssub);
				lhssub.end();
			}
		}
		stat.subtime += clock.getTime()-lasttime;
		if (feas_flag == 1)
		{
			feasbound = feasbound*1.0/nbScens;
			for (int j = 0; j < prob.nbFirstVars; ++j)
				feasbound += xvals[j]*prob.objcoef[j];
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			mastermodel.add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		else
			cout << "Infeasible! Generating feasibility cuts!" << endl;
		lhsaggr.end();
		xvals.end();
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime()-starttime;
		cout << "stat.solvetime = " << stat.solvetime << endl;
		if (stat.solvetime > 10800)
			break;
	}
	subp.suboptcplex.end();
	subp.suboptmodel.end();
	subp.suboptcon.end();
	subp.subopty.end();
	subp.subfeascplex.end();
	subp.subfeasmodel.end();
	subp.subfeascon.end();
	subp.subfeasy.end();
	mastercplex.end();
	mastermodel.end();
	x.end();
}
