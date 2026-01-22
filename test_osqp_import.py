from solvers.osqppurepy import OSQP as OSQP_python

# Test the import
if __name__ == "__main__":
    print(f"Successfully imported OSQP: {OSQP_python}")
    print(f"OSQP class location: {OSQP_python.__module__}")
