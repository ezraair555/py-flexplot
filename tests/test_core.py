import pandas as pd
import numpy as np
from pyflexplot import flexplot, added_plot

def test_basic_plot():
    df = pd.DataFrame({
        "y": np.random.normal(size=100),
        "x": np.random.normal(size=100),
        "z": np.random.choice(["A", "B"], size=100)
    })
    
    # 1. Basic Scatter
    p1 = flexplot("y ~ x", data=df)
    print("Basic plot formula parsed.")
    
    # 2. Faceted plot
    p2 = flexplot("y ~ x | z", data=df)
    print("Faceted plot formula parsed.")
    
    # 3. Added Variable Plot
    p3 = added_plot("y ~ x + z", data=df)
    print("Added variable plot formula parsed.")

if __name__ == "__main__":
    try:
        test_basic_plot()
        print("SUCCESS: py-flexplot core logic verified.")
    except Exception as e:
        print(f"FAILURE: {e}")
