import numpy as np
import pandas as pd
import random


# Set a seed
# random.seed(4)

class Doctor:
    def __init__(self, id, name, specialty, available_days, available_start, available_end,
                 duration):  # , office_names):
        self.doctor_id = id
        self.name = name
        self.specialty = specialty
        self.available_days = available_days
        self.available_start = available_start
        self.available_end = available_end
        self.duration = duration
        # self.office_names = office_names


class DoctorOffice:
    def __init__(self, id, name, specialties, open_hour, close_hour, available_days):
        self.office_id = id
        self.name = name
        self.specialties = specialties
        self.open_hour = open_hour
        self.close_hour = close_hour
        self.available_days = available_days


class Schedule:
    def __init__(self, doctors, buildings, population_size=100, mutation_rate=0.1):
        self.doctors = pd.DataFrame([vars(doc) for doc in doctors])
        # Calculate the number of appointments for each doctor
        self.doctors['Num_Appointments'] = self.doctors.apply(
            lambda row: (row['available_end'] - row['available_start']) / row['duration'] * len(row['available_days']),
            axis=1)
        self.buildings = pd.DataFrame([vars(b) for b in buildings])
        #self.population = []
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.appointments = []
        min_hour = np.min([self.doctors['available_start'], self.doctors['available_end']])
        max_hour = np.max([self.doctors['available_start'], self.doctors['available_end']])
        self.timeslots = np.arange(min_hour, max_hour, 0.5)
        self.fitness_scores = []

    def pick_random_id(self, data_source, column_name):
        if data_source is None or column_name is None:
            raise ValueError("Data source and column name must be specified.")

        unique_ids = data_source[column_name].unique()
        random_id = random.choice(unique_ids)
        return random_id

    def generate_schedule(self):

        # assuming  appointments needed <  maximum capacity
        total_appointments = int(self.doctors['Num_Appointments'].sum())

        # appointments=[]
        for _ in range(self.population_size):
            data = {
                'timeslot_id': np.random.choice(self.timeslots, total_appointments),
                'doctor_id': np.array(
                    [self.pick_random_id(self.doctors, 'doctor_id') for _ in range(total_appointments)]),
                'office_id': np.array(
                    [self.pick_random_id(self.buildings, 'office_id') for _ in range(total_appointments)]),
                # 'week_day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], total_appointments)
                'week_day': np.random.choice(['Monday', "Wednesday", 'Friday'], total_appointments)
            }

            df_appointments = pd.DataFrame(data)
            self.appointments.append(df_appointments)

        #self.population.append(self.appointments)
        # print(self.population[0][0])
        # print('///////////')
        # print(self.appointments[0])
        # self.Mutation(self.appointments[0])
        '''print('p1')
        print(self.appointments[0])
        print('p2')
        print(self.appointments[1])
        a,b=self.Crossover(self.appointments[0],self.appointments[1])
        print('doctores')
        print(self.doctors)'''

    def calculate_fitness(self):

        fitness_values = []
        a = 0
        for individual in self.appointments:

            fitness = 0
            # Check if there are appointments in the same office and date
            count_conflicts = individual.groupby(['timeslot_id', 'week_day', 'office_id']).size().reset_index(
                name='Count')
            filtered_rows = count_conflicts[count_conflicts['Count'] > 1]
            total_conflicts = filtered_rows['Count'].sum()
            fitness += total_conflicts

            # Check if all the appointments for the doctor were scheduled or there are more than expexted

            # Count the number of appointments for each doctor
            appointment_counts = individual['doctor_id'].value_counts().reset_index()
            appointment_counts.columns = ['doctor_id', 'Num_Appointments_Made']
            # check if it really changing nan
            appointment_counts['Num_Appointments_Made'] = appointment_counts['Num_Appointments_Made'].fillna(0)
            '''print('\\\\\\\\\\')
            print(appointment_counts)
            print('\\\\\\\\\\')'''


            df2 = self.doctors.loc[:, ['doctor_id', 'Num_Appointments']]
            df2 = df2.merge(appointment_counts, on='doctor_id', how='left')
            df2['Num_Appointments_Remaining'] = df2['Num_Appointments'] - df2['Num_Appointments_Made']
            df2['Num_Appointments_Remaining'] = df2['Num_Appointments_Remaining'].abs()
            total_appointents_remaining = df2['Num_Appointments_Remaining'].sum()
            fitness += total_appointents_remaining

            # Check if drs are in different places at the same time
            count_conflicts2 = individual.groupby(['timeslot_id', 'week_day', 'doctor_id']).size().reset_index(
                name='Count')
            filtered_rows2 = count_conflicts2[count_conflicts2['Count'] > 1]
            total_conflicts2 = filtered_rows2['Count'].sum()
            fitness += total_conflicts2



            for index, row in individual.iterrows():
                doctor_id = row['doctor_id']
                timeslot_id = row['timeslot_id']
                week_day = row['week_day']
                office_id = row['office_id']

                doctor = self.doctors[self.doctors['doctor_id'] == doctor_id]
                available_start = doctor['available_start'].values[0]
                available_end = doctor['available_end'].values[0]
                available_days = doctor['available_days'].values[0]
                doctor_specialty = doctor['specialty'].values[0]

                office = self.buildings[self.buildings['office_id'] == office_id]
                allowed_specialties = office['specialties'].values[0]

                # Check if timeslot_id falls between available_start and available_end for the doctor
                if not available_start <= timeslot_id < available_end:
                    fitness += 1

                # Check if the weekday doesnt work for the dr
                if not week_day in available_days:
                    fitness += 1
                    # print(f"Doctor {doctor_id}: doesnt work in {week_day} .")

                # check if office is not for all specialties
                if not 0 in allowed_specialties:
                    # Check if assigned matches specialty
                    if not doctor_specialty in allowed_specialties:
                        fitness += 1



            a += 1
            fitness_values.append(fitness)

            # print(f' index: {a} fitness: {fitness}')

        self.fitness_scores = fitness_values
        #print('fitness')
        ##print(self.fitness_scores)

        sorted_appointments = [appointment for _, appointment in sorted(zip(self.fitness_scores, self.appointments), key=lambda x: x[0])]

        #print('appointments')
        #print(self.appointments)
        self.appointments = sorted_appointments
        #print('sorted ')
        #print(self.appointments)
        self.fitness_scores = sorted(self.fitness_scores)
        # print('despues fitnes')
        # print(self.fitness_scores)

    def Crossover(self, parent1, parent2):
        # Make copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        # Get the number of rows and columns
        rows, cols = parent1.shape

        # Randomly select a row and column for crossover
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)

        # Perform crossover
        offspring1.iloc[:row, :] = parent2.iloc[:row, :]
        offspring2.iloc[:row, :] = parent1.iloc[:row, :]

        offspring1.iloc[row, :col + 1] = parent2.iloc[row, :col + 1]
        offspring2.iloc[row, :col + 1] = parent1.iloc[row, :col + 1]

        '''print(f'rows {row}  cols {col}')
        print('off1')
        print(offspring1)
        print('off2')
        print(offspring2)'''

        return offspring1, offspring2

    def Mutation(self, individual):

        for i in range(len(individual)):
            # Determine if each gene should be mutated
            if np.random.random() < self.mutation_rate:
                # Mutate timeslot_id
                individual.at[i, 'timeslot_id'] = np.random.choice(self.timeslots)
            if np.random.random() < self.mutation_rate:
                # Mutate doctor_id
                individual.at[i, 'doctor_id'] = self.pick_random_id(self.doctors, 'doctor_id')
            if np.random.random() < self.mutation_rate:
                # Mutate office_id
                individual.at[i, 'office_id'] = self.pick_random_id(self.buildings, 'office_id')
            if np.random.random() < self.mutation_rate:
                # Mutate week_day
                individual.at[i, 'week_day'] = np.random.choice(['Monday', "Wednesday", 'Friday'])

    def New_Generation(self):

        iterations = 0
        score = -1
        mean_score = -1

        while score != 0 and iterations < 3000:

            next_population = []

            self.calculate_fitness()
            # get the best individual here before while breaks
            score = self.fitness_scores[0]
            mean_score = np.mean(self.fitness_scores)

            # Perform elitism selection 10% of population

            num_rows_10_percent = int(len(self.appointments) * 0.1)
            first_10_percent = self.appointments[:num_rows_10_percent]
            next_population.extend(first_10_percent)

            # use 50% to complete the next generation
            s = int((45 * self.population_size) / 100)
            for _ in range(s):
                parent1 = random.choice(self.appointments[:50])
                parent2 = random.choice(self.appointments[:50])

                a, b = self.Crossover(parent1, parent2)
                self.Mutation(a)
                self.Mutation(b)
                next_population.extend([a, b])

            if iterations % 100 == 0:
                print(
                    f'ite: {iterations} best fitness: {score} mean fitness: {mean_score} popu len: {len(next_population)}')

            self.appointments = next_population
            iterations += 1

        print(f'ite: {iterations} best fitness: {score} mean fitness: {mean_score} popu len: {len(next_population)}')
        #how many got fitness of 0 in the first 5 elements
        zero_values = [x for x in self.fitness_scores[:5] if x == 0]
        rows=len(zero_values)
        print('rows: ',rows)
        #print first x individuals with o score
        print(self.appointments[:rows])

# Create doctors
doctor1 = Doctor(1, "Dr. Smith", "Cardiology", ["Monday", "Friday"], 10.0, 12.0, 0.5)  # , ["Clinic A"])
doctor2 = Doctor(2, "Dr. Johnson", "Dermatology", ["Wednesday"], 10.0, 11.5, 0.5)  # , ["Clinic A"])
doctor3 = Doctor(3,"Dr. Lee", "Neurology", ["Monday","Wednesday"], 12, 13, 0.5)#, ["Clinic B"])#error here
doctor4 = Doctor(4,"Dr. Roberts", "Cardiology", ["Monday", "Friday"], 12, 13.5, 0.5)#, ["Clinic B"])
doctor5 = Doctor(5,"Dr. Rodrigues", "Neurology", ["Monday", "Wednesday"], 13, 14, 0.5)#, ["Clinic A"])
doctor6 = Doctor(6,"Dr. Kim", "Pediatrics", ["Monday"], 10, 11, 0.5)#, ["Clinic B"]) #error here
#doctor7 = Doctor(7,"Dr. Adams", "Gynecology", ["Monday"], 11, 12, 0.5)#, ["Clinic A"])

# Create doctor offices
# office1 = DoctorOffice(1,"Clinic A", ["Cardiology", "Dermatology"], 8.0, 18.0, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
office1 = DoctorOffice(1, "Clinic A", [0], 8.0, 18.0, ["Monday", "Wednesday"])
#office2 = DoctorOffice(2, "Clinic B", ["Pediatrics", "Dermatology"], 8.0, 17.0,["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
office2 = DoctorOffice(2, "Clinic B", ["Cardiology","Pediatrics"], 8.0, 17.0,
                       ["Monday", "Friday"])
# see hw to solve  0=all specialties
# office3 = DoctorOffice(3,"Clinic C", [0], 8.0, 18.0, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

# Create scheduler
# scheduler = Schedule([doctor1, doctor2, doctor3], [office1, office2, office3],2,0.1)
scheduler = Schedule([doctor1, doctor2, doctor3, doctor4, doctor5, doctor6], [office1, office2], 500, 0.1)
scheduler.generate_schedule()
#scheduler.calculate_fitness()
scheduler.New_Generation()