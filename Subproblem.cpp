#include "Subproblem.h"

Subproblem::Subproblem() 
{

}

Subproblem::~Subproblem()
{

}

double Subproblem::subprob(Subprob& subp, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag)
{
	// First try solving the optimization model
	// Set constraint bounds
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		double bd = prob.secondconstrbd[k * prob.nbSecRows + i];
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind < prob.nbFirstVars)
				bd -= prob.CoefMat[i][j] * xvals[ind];
		}
		if (prob.secondconstrsense[i] == -1)
			subp.suboptcon[i].setLB(bd);
		if (prob.secondconstrsense[i] == 0)
		{
			if (subp.suboptcon[i].getUB() < bd)
			{
				subp.suboptcon[i].setUB(bd);
				subp.suboptcon[i].setLB(bd);
			}
			else
			{
				subp.suboptcon[i].setLB(bd);
				subp.suboptcon[i].setUB(bd);
			}
		}
		if (prob.secondconstrsense[i] == 1)
			subp.suboptcon[i].setUB(bd);
	}
	// Set variable bounds
	for (int j = 0; j < prob.nbSecVars; ++j)
	{
		if (prob.secondvarlb[k * prob.nbSecVars + j] != -IloInfinity)
			subp.suboptcon[prob.nbSecRows + j].setLB(prob.secondvarlb[k * prob.nbSecVars + j]);
		else
			subp.suboptcon[prob.nbSecRows + j].setLB(-IloInfinity);
		if (prob.secondvarub[k * prob.nbSecVars + j] != IloInfinity)
			subp.suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-prob.secondvarub[k * prob.nbSecVars + j]);
		else
			subp.suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
	}
	subp.suboptcplex.solve();
	double returnval;
	if (subp.suboptcplex.getStatus() == IloAlgorithm::Optimal)
	{
		returnval = subp.suboptcplex.getObjValue();
		subp.suboptcplex.getDuals(duals, subp.suboptcon);
		feasflag = 1;
	}
	else
	{
		feasflag = 0;
		// infeasible! Get extreme rays
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			double bd = prob.secondconstrbd[k * prob.nbSecRows + i];
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				int ind = prob.CoefInd[i][j];
				if (ind < prob.nbFirstVars)
					bd -= prob.CoefMat[i][j] * xvals[ind];
			}
			if (prob.secondconstrsense[i] == -1)
				subp.subfeascon[i].setLB(bd);
			if (prob.secondconstrsense[i] == 0)
			{
				if (subp.subfeascon[i].getUB() < bd)
				{
					subp.subfeascon[i].setUB(bd);
					subp.subfeascon[i].setLB(bd);
				}
				else
				{
					subp.subfeascon[i].setLB(bd);
					subp.subfeascon[i].setUB(bd);
				}
			}
			if (prob.secondconstrsense[i] == 1)
				subp.subfeascon[i].setUB(bd);
		}
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			if (prob.secondvarlb[k * prob.nbSecVars + j] != -IloInfinity)
				subp.subfeascon[prob.nbSecRows + j].setLB(prob.secondvarlb[k * prob.nbSecVars + j]);
			else
				subp.subfeascon[prob.nbSecRows + j].setLB(-IloInfinity);
			if (prob.secondvarub[k * prob.nbSecVars + j] != IloInfinity)
				subp.subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-prob.secondvarub[k * prob.nbSecVars + j]);
			else
				subp.subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
		}
		subp.subfeascplex.solve();
		subp.subfeascplex.getDuals(duals, subp.subfeascon);
		returnval = subp.subfeascplex.getObjValue();
	}
	return returnval;
}

void Subproblem::construct_second_opt(class IloEnv& env, const TSLP& prob, Subprob& subprob)
{
	subprob.suboptmodel = IloModel(env);
	subprob.suboptcon = IloRangeArray(env);
	subprob.subopty = IloNumVarArray(env, prob.nbSecVars, -IloInfinity, IloInfinity);
	subprob.suboptcplex = IloCplex(subprob.suboptmodel);
	subprob.suboptcplex.setParam(IloCplex::TiLim, 3600);
	subprob.suboptcplex.setParam(IloCplex::Threads, 1);
	subprob.suboptcplex.setParam(IloCplex::SimDisplay, 0);
	subprob.suboptcplex.setParam(IloCplex::BarDisplay, 0);
	subprob.suboptcplex.setParam(IloCplex::EpRHS, 5e-6);
	subprob.suboptcplex.setOut(env.getNullStream());
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += subprob.subopty[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs += subprob.subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -= subprob.subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr suboptobj(env);
	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			suboptobj += subprob.subopty[i - prob.nbFirstVars] * prob.objcoef[i];
	}
	subprob.suboptmodel.add(IloMinimize(env, suboptobj));
	suboptobj.end();
}

void Subproblem::construct_second_feas(class IloEnv& env, const TSLP& prob, Subprob& subprob)
{
	subprob.subfeasmodel = IloModel(env);
	subprob.subfeascon = IloRangeArray(env);
	subprob.subfeasy = IloNumVarArray(env, prob.nbSecVars + prob.nbSecRows, -IloInfinity, IloInfinity);
	subprob.subfeascplex = IloCplex(subprob.subfeasmodel);
	subprob.subfeascplex.setParam(IloCplex::TiLim, 3600);
	subprob.subfeascplex.setParam(IloCplex::Threads, 1);
	subprob.subfeascplex.setParam(IloCplex::SimDisplay, 0);
	subprob.subfeascplex.setParam(IloCplex::BarDisplay, 0);
	subprob.subfeascplex.setOut(env.getNullStream());
	vector<int> extra_ind(prob.nbSecRows, -1);
	for (int j = 0; j < prob.nbSecRows; ++j)
	{
		if (prob.secondconstrsense[j] == -1)
			subprob.subfeasy[prob.nbSecVars + j].setLB(0);
		if (prob.secondconstrsense[j] == 0)
		{
			subprob.subfeasy[prob.nbSecVars + j].setLB(0);
			IloNumVar temp(env, 0, IloInfinity);
			subprob.subfeasy.add(temp);
			extra_ind[j] = subprob.subfeasy.getSize() - 1;
		}
		if (prob.secondconstrsense[j] == 1)
			subprob.subfeasy[prob.nbSecVars + j].setUB(0);
	}
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += subprob.subfeasy[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
		}
		if (prob.secondconstrsense[i] != 0)
			lhs += subprob.subfeasy[prob.nbSecVars + i];
		else
		{
			lhs += subprob.subfeasy[prob.nbSecVars + i];
			lhs -= subprob.subfeasy[extra_ind[i]];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs += subprob.subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -= subprob.subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr subfeasobj(env);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		if (prob.secondconstrsense[i] == -1)
			subfeasobj += subprob.subfeasy[prob.nbSecVars + i];
		if (prob.secondconstrsense[i] == 1)
			subfeasobj -= subprob.subfeasy[prob.nbSecVars + i];
		if (prob.secondconstrsense[i] == 0)
		{
			subfeasobj += subprob.subfeasy[prob.nbSecVars + i];
			subfeasobj += subprob.subfeasy[extra_ind[i]];
		}
	}
	subprob.subfeasmodel.add(IloMinimize(env, subfeasobj));
	subfeasobj.end();
}
