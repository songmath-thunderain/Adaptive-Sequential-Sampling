#include "Subproblem.h"

Subproblem::Subproblem(class IloEnv& env, const TSLP& prob)
{
	//Construct Second Opt
	suboptmodel = IloModel(env);
	suboptcon = IloRangeArray(env);
	subopty = IloNumVarArray(env, prob.nbSecVars, -IloInfinity, IloInfinity);
	suboptcplex = IloCplex(suboptmodel);
	suboptcplex.setParam(IloCplex::TiLim, 3600);
	suboptcplex.setParam(IloCplex::Threads, 1);
	suboptcplex.setParam(IloCplex::SimDisplay, 0);
	suboptcplex.setParam(IloCplex::BarDisplay, 0);
	suboptcplex.setParam(IloCplex::EpRHS, 5e-6);
	suboptcplex.setOut(env.getNullStream());

	//Construct Second Feas
	subfeasmodel = IloModel(env);
	subfeascon = IloRangeArray(env);
	subfeasy = IloNumVarArray(env, prob.nbSecVars + prob.nbSecRows, -IloInfinity, IloInfinity);
	subfeascplex = IloCplex(subfeasmodel);
	subfeascplex.setParam(IloCplex::TiLim, 3600);
	subfeascplex.setParam(IloCplex::Threads, 1);
	subfeascplex.setParam(IloCplex::SimDisplay, 0);
	subfeascplex.setParam(IloCplex::BarDisplay, 0);
	subfeascplex.setOut(env.getNullStream());
}

Subproblem::~Subproblem()
{
	suboptmodel.end(); 
	suboptcplex.end();
	subfeasmodel.end();
	subfeascplex.end();
}

double Subproblem::solve(const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag)
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
			 suboptcon[i].setLB(bd);
		if (prob.secondconstrsense[i] == 0)
		{
			if ( suboptcon[i].getUB() < bd)
			{
				 suboptcon[i].setUB(bd);
				 suboptcon[i].setLB(bd);
			}
			else
			{
				 suboptcon[i].setLB(bd);
				 suboptcon[i].setUB(bd);
			}
		}
		if (prob.secondconstrsense[i] == 1)
			 suboptcon[i].setUB(bd);
	}
	// Set variable bounds
	for (int j = 0; j < prob.nbSecVars; ++j)
	{
		if (prob.secondvarlb[k * prob.nbSecVars + j] != -IloInfinity)
			 suboptcon[prob.nbSecRows + j].setLB(prob.secondvarlb[k * prob.nbSecVars + j]);
		else
			 suboptcon[prob.nbSecRows + j].setLB(-IloInfinity);
		if (prob.secondvarub[k * prob.nbSecVars + j] != IloInfinity)
			 suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-prob.secondvarub[k * prob.nbSecVars + j]);
		else
			 suboptcon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
	}
	 suboptcplex.solve();
	double returnval;
	if ( suboptcplex.getStatus() == IloAlgorithm::Optimal)
	{
		returnval =  suboptcplex.getObjValue();
		 suboptcplex.getDuals(duals,  suboptcon);
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
				 subfeascon[i].setLB(bd);
			if (prob.secondconstrsense[i] == 0)
			{
				if ( subfeascon[i].getUB() < bd)
				{
					 subfeascon[i].setUB(bd);
					 subfeascon[i].setLB(bd);
				}
				else
				{
					 subfeascon[i].setLB(bd);
					 subfeascon[i].setUB(bd);
				}
			}
			if (prob.secondconstrsense[i] == 1)
				 subfeascon[i].setUB(bd);
		}
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			if (prob.secondvarlb[k * prob.nbSecVars + j] != -IloInfinity)
				 subfeascon[prob.nbSecRows + j].setLB(prob.secondvarlb[k * prob.nbSecVars + j]);
			else
				 subfeascon[prob.nbSecRows + j].setLB(-IloInfinity);
			if (prob.secondvarub[k * prob.nbSecVars + j] != IloInfinity)
				 subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-prob.secondvarub[k * prob.nbSecVars + j]);
			else
				 subfeascon[prob.nbSecRows + prob.nbSecVars + j].setLB(-IloInfinity);
		}
		 subfeascplex.solve();
		 subfeascplex.getDuals(duals,  subfeascon);
		returnval =  subfeascplex.getObjValue();
	}
	return returnval;
}

void Subproblem::construct_second_opt(class IloEnv& env, const TSLP& prob)
{
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs +=  subopty[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 suboptcon.add(range);
		 suboptmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs +=  subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 suboptcon.add(range);
		 suboptmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -=  subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 suboptcon.add(range);
		 suboptmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr suboptobj(env);
	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			suboptobj +=  subopty[i - prob.nbFirstVars] * prob.objcoef[i];
	}
	 suboptmodel.add(IloMinimize(env, suboptobj));
	suboptobj.end();
}

void Subproblem::construct_second_feas(class IloEnv& env, const TSLP& prob)
{
	vector<int> extra_ind(prob.nbSecRows, -1);
	for (int j = 0; j < prob.nbSecRows; ++j)
	{
		if (prob.secondconstrsense[j] == -1)
			 subfeasy[prob.nbSecVars + j].setLB(0);
		if (prob.secondconstrsense[j] == 0)
		{
			 subfeasy[prob.nbSecVars + j].setLB(0);
			IloNumVar temp(env, 0, IloInfinity);
			 subfeasy.add(temp);
			extra_ind[j] =  subfeasy.getSize() - 1;
		}
		if (prob.secondconstrsense[j] == 1)
			 subfeasy[prob.nbSecVars + j].setUB(0);
	}
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs +=  subfeasy[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
		}
		if (prob.secondconstrsense[i] != 0)
			lhs +=  subfeasy[prob.nbSecVars + i];
		else
		{
			lhs +=  subfeasy[prob.nbSecVars + i];
			lhs -=  subfeasy[extra_ind[i]];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 subfeascon.add(range);
		 subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs +=  subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 subfeascon.add(range);
		 subfeasmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -=  subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		 subfeascon.add(range);
		 subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr subfeasobj(env);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		if (prob.secondconstrsense[i] == -1)
			subfeasobj +=  subfeasy[prob.nbSecVars + i];
		if (prob.secondconstrsense[i] == 1)
			subfeasobj -=  subfeasy[prob.nbSecVars + i];
		if (prob.secondconstrsense[i] == 0)
		{
			subfeasobj +=  subfeasy[prob.nbSecVars + i];
			subfeasobj +=  subfeasy[extra_ind[i]];
		}
	}
	 subfeasmodel.add(IloMinimize(env, subfeasobj));
	subfeasobj.end();
}
