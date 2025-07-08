import numpy as np
from typing import Optional, Union

class TimeWeightedEMA:
    """
    Time-weighted Exponential Moving Average that handles irregular time intervals.
    
    The EMA is computed as:
    EMA_t = value_t * (1 - decay_factor) + EMA_{t-1} * decay_factor
    
    Where decay_factor = exp(-alpha * delta_t)
    """
    
    def __init__(self, 
                 half_life: Optional[float] = None,
                 decay_rate: Optional[float] = None,
                 smoothing_factor: Optional[float] = None):
        """
        Initialize with one of:
        - half_life: time for EMA to decay to 50% weight
        - decay_rate: alpha parameter (decay per unit time)
        - smoothing_factor: traditional EMA parameter for unit time interval
        """
        if sum(x is not None for x in [half_life, decay_rate, smoothing_factor]) != 1:
            raise ValueError("Specify exactly one of: half_life, decay_rate, smoothing_factor")
        
        if half_life is not None:
            self.alpha = np.log(2) / half_life
        elif decay_rate is not None:
            self.alpha = decay_rate
        else:  # smoothing_factor
            # Convert traditional EMA parameter to decay rate
            # smoothing_factor = 2/(n+1) where n is the period
            self.alpha = -np.log(1 - smoothing_factor)
        
        self.ema = None
        self.last_time = None
    
    def update(self, value: float, timestamp: float) -> float:
        """
        Update EMA with new value at given timestamp.
        
        Args:
            value: New observation
            timestamp: Time of observation (any monotonic unit)
            
        Returns:
            Updated EMA value
        """
        if self.ema is None:
            # First value - initialize
            self.ema = value
            self.last_time = timestamp
        else:
            # Calculate time since last update
            delta_t = timestamp - self.last_time
            
            if delta_t < 0:
                raise ValueError("Timestamp must be >= last timestamp")
            
            if delta_t > 0:
                # Calculate decay factor based on time elapsed
                decay_factor = np.exp(-self.alpha * delta_t)
                
                # Update EMA
                self.ema = value * (1 - decay_factor) + self.ema * decay_factor
                self.last_time = timestamp
            # If delta_t == 0, keep current EMA unchanged
        
        return self.ema
    
    def get_value(self) -> Optional[float]:
        """Get current EMA value."""
        return self.ema
    
    def reset(self):
        """Reset the EMA."""
        self.ema = None
        self.last_time = None