/*
   Functions used to execute option 1 in the main function.
   *****Insert description of option 1.*****


*/

// Only solve_level is different from option 0

if (option == 1)
{
  // Level method
  vector<int> samples;
  for (int k = 0; k < prob.nbScens; ++k)
    samples.push_back(k);
  VectorXf xiterateXf(prob.nbFirstVars);
  solve_level(env, prob, stat, clock, samples, xiterateXf, 0);
  cout << "relaxobjval = " << stat.relaxobjval << endl;
  cout << "feasobjval = " << stat.feasobjval << endl;
  cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
}

void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option)
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
	IloEnv meanenv;
	IloNumArray meanxvals(meanenv);
	double meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
	vector<double> xiterate(prob.nbFirstVars);
	// Assign xiterate to meanx
	for (int j = 0; j < prob.nbFirstVars; ++j)
		xiterate[j] = meanxvals[j];
	bool feas_flag = 1;
	meanxvals.end();
	meanenv.end();

	// Define Master program
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
	obj += theta;
	mastermodel.add(IloMinimize(env, obj));
	obj.end();
	IloCplex mastercplex(mastermodel);
	mastercplex.setParam(IloCplex::TiLim,10800);
	mastercplex.setParam(IloCplex::Threads, 1);
	mastercplex.setParam(IloCplex::BarDisplay, 0);
	mastercplex.setParam(IloCplex::SimDisplay, 0);
	mastercplex.setOut(env.getNullStream());
	// Define Level quadratic program
	IloEnv lenv;
	IloModel levelmodel(lenv);
	IloNumVarArray lx(lenv, prob.firstvarlb, prob.firstvarub);
	IloNumVar ltheta(lenv, -IloInfinity, IloInfinity);
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(lenv);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += lx[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(lenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		levelmodel.add(range);
		lhs.end();
	}
	// define the first constraint for level master
	IloExpr lsum(lenv);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		lsum += lx[j]*prob.objcoef[j];
	lsum += ltheta;
	IloRange rangeub(lenv, -IloInfinity, lsum, IloInfinity);
	levelmodel.add(rangeub);
	lsum.end();

	IloObjective lobj = IloMinimize(lenv);
	levelmodel.add(lobj);

	IloCplex levelcplex(levelmodel);
	levelcplex.setParam(IloCplex::TiLim,10800);
	levelcplex.setParam(IloCplex::Threads, 1);
	levelcplex.setParam(IloCplex::BarDisplay, 0);
	levelcplex.setOut(env.getNullStream());

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	int nbScens = samples.size();

	// This part needs to be changed into Level method iteration
	while ((stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > accuracy || feas_flag == 0)
	{
		feas_flag = 1;
		IloEnv env2;
		stat.iter++;
		IloNumArray xiteratevals(env2, prob.nbFirstVars);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			xiterateXf(j) = xiterate[j];
			xiteratevals[j] = xiterate[j];
		}
		double feasbound = 0.0;
		for (int j = 0; j < prob.nbFirstVars; ++j)
			feasbound += xiterate[j]*prob.objcoef[j];

		IloExpr lhsaggr(env);
		lhsaggr += theta;
		IloExpr llhsaggr(lenv);
		llhsaggr += ltheta;
		double rhsaggr = 0;
		double lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each partition
			IloNumArray duals(env2);
			bool feasflag;
			double subobjval = subprob(subp, prob, xiteratevals, duals, samples[k], feasflag);
			VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
				double sum_xvals = opt_cut_coef.dot(xiterateXf);
				double rhssub = subobjval + sum_xvals;
				feasbound += subobjval*1.0/nbScens;
				rhsaggr += rhssub*1.0/nbScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
						llhsaggr += lx[j]*opt_cut_coef[j]*1.0/nbScens;
					}
				}

			}
			else
			{
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				IloExpr llhssub(lenv);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
					{
						llhssub += lx[j]*feas_cut_coef[j];
						lhssub += x[j]*feas_cut_coef[j];
					}
				}
				double rhssub = sum_xvals+subobjval;
				stat.num_feas_cuts++;
				mastermodel.add(lhssub >= rhssub);
				levelmodel.add(llhssub >= rhssub);
				lhssub.end();
				llhssub.end();
			}
		}
		xiteratevals.end();
		stat.subtime += clock.getTime()-lasttime;
		if (feas_flag == 1)
		{
			//cout << "feasbound = " << feasbound << ", feasobjval = " << stat.feasobjval << endl;
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			levelmodel.add(llhsaggr >= rhsaggr);
			mastermodel.add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		lhsaggr.end();
		llhsaggr.end();
		// Now solve the master, get a lower bound
		lasttime = clock.getTime();
		mastercplex.solve();
		stat.relaxobjval = mastercplex.getObjValue();
		stat.mastertime += clock.getTime()-lasttime;

		// Now solve the qp level problem
		// update the upper bound, (1-\lambda)F^{k+1}+\lambda F^*
		rangeub.setUB(0.5*stat.relaxobjval + 0.5*stat.feasobjval);
		IloExpr objExpr(lenv);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			objExpr += lx[j]*lx[j];
			objExpr -= lx[j]*2*xiterate[j];
		}
		lobj.setExpr(objExpr);
		objExpr.end();

		double startqptime = clock.getTime();
		levelcplex.solve();
		stat.qptime += clock.getTime()-startqptime;
		IloNumArray lxval(lenv);
		levelcplex.getValues(lxval, lx);
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterate[j] = lxval[j];
		lxval.end();
		env2.end();
		//cout << "relaxobjval = " << stat.relaxobjval << endl;
		//cout << "feasobjval = " << stat.feasobjval << endl;
		//cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime()-starttime;
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
	levelcplex.end();
	levelmodel.end();
	lx.end();
	x.end();
}
