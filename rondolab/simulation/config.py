from pydantic import BaseModel, Field, model_validator


class Config(BaseModel):
    width: float = Field(15, ge=10, le=25)
    height: float = Field(15, ge=10, le=25)
    dt: float = Field(.05, ge=.01, le=.05)
    decision_steps: int = Field(4, ge=1, le=10)
    duration: float = Field(40, ge=5, le=180)
    player_speed: float = Field(3.2, ge=.1, le=7)
    defender_speed: float = Field(2.3, ge=0, le=7)
    acceleration: float = Field(6, ge=1, le=15)
    turning_rate: float = Field(5.5, ge=.5, le=12)
    ball_speed: float = Field(10, ge=4, le=18)
    resistance: float = Field(.3, ge=0, le=2)
    pass_error: float = Field(.015, ge=0, le=.3)
    control_radius: float = Field(.65, ge=.3, le=1)
    interception_radius: float = Field(.6, ge=.2, le=1)
    reaction_time: float = Field(.15, ge=0, le=.6)
    support_radius: float = Field(1.3, ge=0, le=3)
    max_hold: float = Field(3, ge=.4, le=10)
    max_touches: int = Field(0, ge=0, le=2)
    pair_limit: int = Field(0, ge=0, le=8)
    defender_strategy: str = 'cover_press'
    reward: str = 'moderate'
    loss_penalty: float = Field(2, ge=0, le=10)
    curriculum: int = Field(4, ge=1, le=6)

    @model_validator(mode='after')
    def choices(self):
        if self.defender_strategy not in ('chase', 'nearest', 'lanes', 'cover_press'):
            raise ValueError('Unknown defending strategy')
        if self.reward not in ('sparse', 'moderate', 'heavy'):
            raise ValueError('Unknown reward configuration')
        return self


PRESETS = {
    'Classic 6v2': {},
    'Beginner Rondo': {'defender_speed': 1, 'defender_strategy': 'chase', 'curriculum': 2},
    'High Pressure': {'defender_speed': 3.4, 'max_hold': 1.5, 'curriculum': 5},
    'One-Touch Rondo': {'max_touches': 1, 'max_hold': .7},
    'Adaptive Defenders': {'defender_strategy': 'lanes', 'defender_speed': 2.7},
    'Competitive Self-Play': {'curriculum': 6},
}
