## Doctor Appointment Scheduling Algorithm

This Python script is designed to create a scheduling algorithm for doctor appointments. The main classes in the script are:

### 1. `Doctor` Class
   - Represents a doctor with attributes such as:
     - `doctor_id`: Unique identifier for the doctor.
     - `name`: Name of the doctor.
     - `specialty`: Medical specialty of the doctor.
     - `available_days`: Days of the week the doctor is available.
     - `available_start`: Starting time of availability.
     - `available_end`: Ending time of availability.
     - `duration`: Duration of each appointment.
   
### 2. `DoctorOffice` Class
   - Represents a doctor's office with attributes such as:
     - `office_id`: Unique identifier for the office.
     - `name`: Name of the office.
     - `specialties`: List of medical specialties available at the office.
     - `open_hour`: Opening hour of the office.
     - `close_hour`: Closing hour of the office.
     - `available_days`: Days of the week the office is open.
   
### 3. `Schedule` Class
   - Manages the scheduling algorithm and optimization process.
   - Key methods include:
     - `generate_schedule()`: Generates initial random schedules.
     - `calculate_fitness()`: Calculates the fitness score of each schedule.
     - `Crossover()`: Performs crossover between two schedules.
     - `Mutation()`: Mutates the genes of a schedule.
     - `New_Generation()`: Creates a new generation of schedules based on fitness.

### Main Functionality
1. **Initialization:**
   - Doctors and their availability, as well as doctor offices, are defined.
   - A `Schedule` object is created with a list of doctors and offices.

2. **Generating Schedule:**
   - The algorithm generates random schedules based on doctor availability and office constraints.
   - Each schedule includes appointment details such as time slot, doctor ID, office ID, and weekday.

3. **Fitness Calculation:**
   - The fitness score of each schedule is calculated based on various criteria:
     - Avoiding conflicts in the same office and time slot.
     - Ensuring doctors have the correct number of appointments.
     - Checking if doctors are scheduled at different places simultaneously.
     - Verifying if appointments align with doctor availability and specialties.

4. **Evolutionary Algorithm:**
   - The algorithm evolves over generations, aiming to improve fitness scores.
   - Elitism selection retains the top-performing schedules.
   - Crossover and mutation create new schedules from selected parents.
   - The process continues for a set number of iterations or until a fitness threshold is reached.

### Example Execution:
```python
# Create doctors and offices
doctor1 = Doctor(1, "Dr. Smith", "Cardiology", ["Monday", "Friday"], 10.0, 12.0, 0.5)
doctor2 = Doctor(2, "Dr. Johnson", "Dermatology", ["Wednesday"], 10.0, 11.5, 0.5)
doctor3 = Doctor(3, "Dr. Lee", "Neurology", ["Monday", "Wednesday"], 12, 13, 0.5)
office1 = DoctorOffice(1, "Clinic A", [0], 8.0, 18.0, ["Monday", "Wednesday"])
office2 = DoctorOffice(2, "Clinic B", ["Cardiology", "Pediatrics"], 8.0, 17.0, ["Monday", "Friday"])

# Create scheduler and run scheduling algorithm
scheduler = Schedule([doctor1, doctor2, doctor3], [office1, office2], 500, 0.1)
scheduler.generate_schedule()
scheduler.New_Generation()
