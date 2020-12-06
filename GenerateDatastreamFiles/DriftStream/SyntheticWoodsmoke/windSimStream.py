import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.colors
import numpy as np
import math
import random
import statistics
import argparse
import cProfile, pstats, io

class PollutionSource:
    def __init__(self, x, y, strength, expansion):
        self.x = x
        self.y = y
        self.strength = strength

        self.initial_radius = expansion * 10
        self.expansion = expansion

        self.last_strength = strength
        self.period = np.random.randint(0, 360)

        self.last_emitted = None

    def emit(self):
        self.period += np.random.randint(1, 36)
        # s = self.strength * (math.sin(math.radians(self.period)) * 0.5 + (np.random.rand() * 0.5 - 0.25) + 1)
        s = self.strength * (math.sin(math.radians(self.period)) * 0.1 + 1)
        # print(s)
        # s = self.strength
        e = PollutionEmission(self.x, self.y, self.initial_radius, s, self.expansion)
        if not self.last_emitted is None and ((self.x - self.last_emitted.x) * (self.x - self.last_emitted.x) + (self.y - self.last_emitted.y) * (self.y - self.last_emitted.y)) < self.last_emitted.r:
            self.last_emitted.child = e
        else:
            e.first_emitted = True
        self.last_emitted = e
        return e

class PollutionEmission:
    def __init__(self, x, y, r, initial_strength, expansion):
        self.x = x
        self.y = y
        self.r = r
        self.strength = initial_strength

        self.expansion = expansion
        # self.diminish = max(2, self.strength * 0.02)
        self.diminish = 0.05

        self.alive = True

        self.child = None
        self.first_emitted = False
        self.last_time_step = -1
    
    def propagate(self, wind_vec, ts):
        if ts == self.last_time_step:
            return
        if self.alive:
            self.x += wind_vec[0]
            self.y += wind_vec[1]

            self.r += self.expansion
            # self.strength -= self.diminish
            self.strength *= (1- self.diminish)

        if self.strength <= 10:
            self.alive = False
            if not self.child is None:
                self.child.first_emitted = True
        self.last_time_step = ts
            


class WindSimGenerator:
    
    def __init__(self, concept = 0, produce_image = False, num_sensors = 8, sensor_pattern = 'circle', sample_random_state = None, x_trail = 0):
        self.anchor = [0, 0]

        # Treat as meters, i.e 1000 = a 1000x1000 m window
        self.window_width = 200

        # How many grid squares. window_width / window_divisions = size of grid square in meters.
        self.window_divisions = 200

        self.grid_square_width = self.window_width / self.window_divisions

        self.wind_direction = None
        self.concept = concept
        self.set_wind(concept, strength=2.2)
        
        # Scale of noise, bigger = bigger noise features.
        self.noise_scale = 5000
        self.sensor_locs = [(4286, 1995), (734, 773), (1949, 1462), (1926, 2479), (3219, 1758),
                        (4218, 3532), (3604, 1469), (3676, 2798), (714, 1654), (1158, 3263),
                        (2947, 4227), (2515, 3419), (2865, 2577)]
        self.sensor_square_locs = []
        for sx, sy in self.sensor_locs:
            self.sensor_square_locs.append((int(sx / self.grid_square_width), int(sy / self.grid_square_width)))
        #print(self.sensor_square_locs)

        center_sensor_loc = (int(self.window_width / 2), int(self.window_width / 2))
        self.optimal_sensor_locs = [center_sensor_loc]
        if sensor_pattern == 'circle':
            radius = 678
            angle = 0
            while angle < 360:
                px = center_sensor_loc[0] + math.cos(math.radians(angle)) * radius
                py = center_sensor_loc[1] + math.sin(math.radians(angle)) * radius
                self.optimal_sensor_locs.append((px, py))
                angle += 360 / num_sensors
        else:
            num_sensors_x = math.ceil(math.sqrt(num_sensors))
            num_sensors_y = math.ceil(math.sqrt(num_sensors))
            sensor_x_gap = self.window_width / (num_sensors_x + 1)
            sensor_y_gap = self.window_width / (num_sensors_y + 1)

            for c in range(num_sensors_x):
                for r in range(num_sensors_y):
                    px = (c+1) * sensor_x_gap
                    py = (r+1) * sensor_y_gap
                    self.optimal_sensor_locs.append((px, py))




        # print(self.optimal_sensor_locs)
        self.optimal_sensor_square_locs = []
        for sx, sy in self.optimal_sensor_locs:
            self.optimal_sensor_square_locs.append((int(sx / self.grid_square_width), int(sy / self.grid_square_width)))
        
        # print(self.optimal_sensor_square_locs)

        # Timestep in seconds
        self.timestep = 60 * 10

        self.produce_image = produce_image
        self.last_update_image = None

        self.emitted_values = []
        # The number of timesteps a prediction is ahead of X.
        # I.E the y value received with a given X is the y value
        # y_lag ahead of the reveived X values.
        # For this sim, it should be 10 minutes.
        self.y_lag = 1
        self.x_trail = x_trail

        self.prepared = False

        self.world = np.zeros(shape = (self.window_width, self.window_width), dtype=float)
        self.sources = []
        self.pollution = []
        self.pollution_chain = []

        if sample_random_state is None:
            self.concept = np.random.randint(0, 1000)
        else:
            self.concept = sample_random_state

        # self.set_concept(self.concept)
        self.ex = 0

        self.n_targets = 1

        
    def set_concept(self, concept_seed):
        concept_generator = np.random.RandomState(concept_seed)

        self.wind_direction = concept_generator.randint(0, 360)
        self.wind_strength = ((concept_generator.rand() * 60) + 10) / (self.window_width / 5)
        wind_direction_corrected = (self.wind_direction - 90) % 360
        self.wind_direction_radians = math.radians(wind_direction_corrected)

        self.wind_strength_x = math.cos(self.wind_direction_radians) * self.wind_strength
        self.wind_strength_y = math.sin(self.wind_direction_radians) * self.wind_strength

        self.sources = []

        num_sources = concept_generator.randint(10, 20)
        for s in range(num_sources):
            x = concept_generator.randint(0, self.window_width)
            y = concept_generator.randint(0, self.window_width)
            strength = concept_generator.randint(10, 255)
            strength = 170
            size = concept_generator.randint(1, 4)
            self.sources.append(PollutionSource(x, y, strength, (self.window_width / 750) * size))
        

    def get_direction_from_concept(self, concept):
        return 45 * concept

    def set_wind(self, concept = 0, direc = None, strength = None):
        self.concept = concept
        wind_direction = self.get_direction_from_concept(concept)
        if direc != None:
            wind_direction = direc
        if wind_direction == self.wind_direction:
            return
        # In knots: 1 knot = 0.514 m/s
        # Data average is around 2.2
        self.wind_strength = strength if strength != None else self.wind_strength
        self.wind_direction = wind_direction % 360

        # Wind direction is a bearing, want a unit circle degree.
        wind_direction_corrected = (self.wind_direction - 90) % 360
        self.wind_direction_radians = math.radians(wind_direction_corrected)

        self.wind_strength_x = math.cos(self.wind_direction_radians) * self.wind_strength
        self.wind_strength_y = math.sin(self.wind_direction_radians) * self.wind_strength

    def update(self):
        self.ex += 1
        self.anchor[0] += self.wind_strength_x * 0.514 * self.timestep
        self.anchor[1] += self.wind_strength_y * 0.514 * self.timestep

        alive_p = []
        alive_p_first = []
        for p in self.pollution_chain:
            p.propagate((self.wind_strength_x, self.wind_strength_y), self.ex)
            if p.alive:
                # alive_p.append(p)
                if p.first_emitted:
                    alive_p_first.append(p)
            while not p.child is None:
                p = p.child
                p.propagate((self.wind_strength_x, self.wind_strength_y), self.ex)
                if p.alive:
                    # alive_p.append(p)
                    if p.first_emitted:
                        alive_p_first.append(p)
        # self.pollution = alive_p
        self.pollution_chain = alive_p_first

        if self.ex % 10 == 0:
            for s in self.sources:
                emission = s.emit()
                if emission.first_emitted:
                    self.pollution_chain.append(emission)
                # self.pollution.append(emission)

        if self.produce_image:
            z = self.get_full_image()
        else:
            z = None


        sensor_windows = []
        for x, y in self.optimal_sensor_locs:
            sensor_collection = []
            sensor_sum = 0
            look_at_sensors = []
            for p in self.pollution_chain:
                node = p
                if ((x - p.x) ** 2) + ((y - p.y) ** 2) > (p.r**2 + 4):
                    continue
                else:
                    keep_looking = True
                    look_at_sensors.append(p)
                    while not node.child is None and keep_looking:
                        node = node.child
                        dist = ((x - node.x) ** 2) + ((y - node.y) ** 2) > (node.r**2 + 4)
                        if dist:
                            keep_looking = False
                            break
                        look_at_sensors.append(node)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    noise_x_pos = int((x + dx))
                    noise_y_pos = int((y + dy))
                    if self.produce_image and False:
                        sensor_collection.append(z[noise_y_pos, noise_x_pos])
                        sensor_sum += z[noise_y_pos, noise_x_pos]
                    else:
                        value = 0
                        for p in look_at_sensors:
                            if ((noise_x_pos - p.x) ** 2) + ((noise_y_pos - p.y) ** 2) > p.r**2:
                                continue
                            else:
                                value += p.strength
                        # sensor_collection.append(value)
                        sensor_sum += value
            # sensor_windows.append((statistics.mean(sensor_collection),
            #                         statistics.stdev(sensor_collection),
            #                         sum(sensor_collection)))
            sensor_windows.append(sensor_sum)

        if self.produce_image:
            for sx, sy in self.optimal_sensor_square_locs:
                sensor_collection = []
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        z[sy + dy, sx + dx] = 255

            self.last_update_image = z
        
        return (list(map(lambda x: x, sensor_windows[1:])), sensor_windows[0])

    def get_last_image(self):
        return self.last_update_image
    
    def get_full_image(self):
        # self.world = np.zeros(shape = (self.window_width, self.window_width), type=float)
        x_rolls = round(self.wind_strength_x)
        y_rolls = round(self.wind_strength_y)

        self.world = np.roll(self.world, x_rolls, axis = 1)
        self.world = np.roll(self.world, y_rolls, axis = 0)

        
        z = np.copy(self.world)
        for p in self.pollution_chain:
            while not p is None:
                for x in range(round(p.x - p.r), round(p.x + p.r)):
                    for y in range(round(p.y - p.r), round(p.y + p.r)):
                        if x < 0 or x >= self.window_width:
                            continue
                        if y < 0 or y >= self.window_width:
                            continue
                        if ((x - p.x) ** 2) + ((y - p.y) ** 2) > p.r**2:
                            continue
                        z[x, y] += p.strength
                p = p.child
            
        return z
    
    def add_emissions(self):
        X,y = self.update()
        for index, emit in enumerate([y] + X):
            if index >= len(self.emitted_values):
                self.emitted_values.append([])
            self.emitted_values[index].append(emit)

    def prepare_for_use(self):

        # Need to set up y values for X values y_lag behind.
        print("prepared")
        if self.wind_direction is None:
            self.set_concept(self.concept)
        self.add_emissions()
        for i in range(1 + self.x_trail + self.y_lag + 500):
            self.add_emissions()
        self.X_index = 1 + self.x_trail + 500
        self.y_index = self.X_index + self.y_lag
        self.prepared = True
        
    
    def next_sample(self, batch_size = 1):
        
        # X = list(map(lambda x: x[self.X_index], self.emitted_values[1:]))
        x_vals = []
        y_vals = []
        for b in range(batch_size):
            self.add_emissions()
            X = []
            for i, x_emissions in enumerate(self.emitted_values[1:]):
                index = i + 1
                for x_i in range(self.X_index - self.x_trail, self.X_index + 1):
                    X.append(x_emissions[x_i])
                # current_x = x_emissions[self.X_index]
                # last_x = x_emissions[self.X_index - 1]
                # X.append(current_x)
                # X.append(1 if current_x > last_x else 0)
            current_y = self.emitted_values[0][self.y_index]
            
            last_y = self.emitted_values[0][self.y_index - 1]
            # print(f"{current_y} <- {last_y}")
            y = 1 if current_y > last_y else 0
            self.X_index += 1
            self.y_index += 1
            x_vals.append(X)
            y_vals.append(y)
            # y = [y]
        self.current_sample_x = np.array(x_vals)
        self.current_sample_y = np.array(y_vals)
        self.n_features = len(x_vals[0])
        self.n_num_features = len(x_vals[0])
        self.n_cat_features = 0
        self.n_classes = 2
        self.cat_features_idx = []
        self.sample_idx = 0
        self.feature_names = None
        self.target_names = None
        self.target_values = None
        self.name = "wind"
        return (np.array(x_vals), np.array(y_vals))
    
    def get_info(self, concept = None, strength = None):
        c = concept if concept != None else self.concept
        s = strength if strength != None else self.wind_strength
        return f"WIND: Direction: {self.get_direction_from_concept(c)}, Speed: {s}"
    
    def n_remaining_samples(self):
        return 100
    
    def n_samples(self):
        return 100


    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ns", "--nsensor", type=int,
        help="Number of sensors", default=8)
    ap.add_argument("-st", "--sensortype",
        help="How sensors are arranged", default="circle", choices=["circle", "grid"])
    args = vars(ap.parse_args())

    Writer = animation.writers['pillow']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=900)
    fig = plt.figure()
    ax = plt.gca()
    plt.gray()
    n_concepts = 5
    concepts = []
    for c in range(n_concepts):
        concepts.append(np.random.randint(0, 1000))
    current_concept = 0
    stream = WindSimGenerator(produce_image=True, num_sensors= args['nsensor'], sensor_pattern=args['sensortype'])
    stream.prepare_for_use()
    stream.set_concept(current_concept % n_concepts)
    count = 0
    drift_count = 100
    # pr = cProfile.Profile()
    # pr.enable()

    # for ex in range(100):
    #     X,y = stream.next_sample()
    
    # pr.disable()
    # s = io.StringIO()
    # # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s)
    # ps.print_stats()
    # print(s.getvalue())
    def animate(i):
        global count
        global current_concept
        X, y = stream.next_sample()
        print(f"X: {X}, y: {y}")
        if stream.produce_image:
            z = stream.get_last_image()
            #print(f"X: {X}, y: {y}")
            # Plot the grid
            plt.clf()
            plt.imshow(z, norm = matplotlib.colors.Normalize(0, 255))
        count += 1
        if count >= drift_count:
            current_concept += 1
            stream.set_concept(current_concept % n_concepts)
            count = 0
    ani = animation.FuncAnimation(fig, animate, init_func = lambda: [], repeat=True)
    plt.show()


# closest_match = None
# closest_distance = None
# if last_sensor_windows != None:

#     for i,sw in enumerate(last_sensor_windows[1:]):
#         distance = np.sqrt(np.sum(np.power(np.array(sw) - np.array(sensor_windows[0]), 2)))
#         if closest_distance == None or distance < closest_distance:
#             closest_distance = distance
#             closest_match = i + 1
#     print(f"Closest match to center was {closest_match}")
# last_sensor_windows = sensor_windows
# for si, l in enumerate(optimal_sensor_square_locs):
#     sx, sy = l
#     for dx in range(-3, 4):
#         for dy in range(-3, 4):
#             if si in [closest_match, 0]:
#                 z[sy + dy, sx + dx] = 1