from pyomo.common.errors import ApplicationError
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    SolverFactory,
    Var,
    minimize,
    value,
)


def main() -> None:
    model = ConcreteModel()

    model.x = Var(initialize=0.0)
    model.y = Var(initialize=0.0)

    model.obj = Objective(
        expr=(model.x - 1.0) ** 2 + (model.y - 2.0) ** 2,
        sense=minimize,
    )
    model.constraint = Constraint(expr=model.x + model.y >= 1.0)

    solver = SolverFactory("ipopt")

    print(f"IPOPT available: {solver.available(exception_flag=False)}")
    print(f"IPOPT executable: {solver.executable()}")
    print("Starting solve...\n")

    try:
        results = solver.solve(model, tee=True)
    except ApplicationError as exc:
        print("\nSolve failed.")
        print(f"IPOPT exited abnormally: {exc}")
        return

    print("\nSolve finished.")
    print(f"Solver status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")
    print(f"x = {value(model.x):.6f}")
    print(f"y = {value(model.y):.6f}")
    print(f"objective = {value(model.obj):.6f}")


if __name__ == "__main__":
    main()
