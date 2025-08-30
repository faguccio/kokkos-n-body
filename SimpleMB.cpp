#include <iostream>
#include <cmath>
#include <fstream>
#include <random>

static const int MAX_PARTICLES = 100;

struct vec3 {
    double x, y, z;

    vec3(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {
    }

    constexpr vec3 operator+(const vec3& other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }

    constexpr vec3 operator-(const vec3& other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }

    constexpr vec3 operator*(double scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }

    constexpr vec3& operator+=(const vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr double dot(const vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    constexpr double magnitude() const {
        return sqrt(x * x + y * y + z * z);
    }

    constexpr double magnitudeSquared() const {
        return x * x + y * y + z * z;
    }
};

struct Particle {
    vec3 pos, vel, acc, force;
    double mass;

    Particle() : mass(1.0) {
    }

    Particle(const vec3& position, const vec3& velocity, double m)
        : pos(position), vel(velocity), mass(m) {
    }
};


class ManyBodySimulation {
private:
    Particle particles[MAX_PARTICLES];
    int particlesCount = 0;
    double dt;
    double G;
    double softening;

    void resetForces() {
        for (int i = 0; i < particlesCount; ++i) {
            particles[i].force = vec3(0.0, 0.0, 0.0);
        }
    }

public:
    ManyBodySimulation(double timestep = 0.01, double gravitational_constant = 1.0,
                       double soft = 0.1)
        : dt(timestep), G(gravitational_constant), softening(soft) {
    }

    bool addParticle(const Particle& p) {
        if (particlesCount >= MAX_PARTICLES) {
            return false;
        }
        particles[particlesCount++] = p;
        return true;
    }

    void initializeRandomParticles(int n, double mass_range = 1.0, double pos_range = 10.0,
                                   double vel_range = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> pos_dis(-pos_range, pos_range);
        std::uniform_real_distribution<> vel_dis(-vel_range, vel_range);
        std::uniform_real_distribution<> mass_dis(0.1, mass_range);

        particlesCount = 0;

        int particles_to_create = std::min(n, MAX_PARTICLES);

        for (int i = 0; i < particles_to_create; ++i) {
            vec3 position(pos_dis(gen), pos_dis(gen), pos_dis(gen));
            vec3 velocity(vel_dis(gen), vel_dis(gen), vel_dis(gen));
            double mass = mass_dis(gen);

            Particle p(position, velocity, mass);
            addParticle(p);
        }
    }


    void calculateForces() {
        resetForces();
        for (int i = 0; i < particlesCount; ++i) {
            for (int j = i + 1; j < particlesCount; ++j) {
                Particle& pi = particles[i];
                Particle& pj = particles[j];

                // Distance vector
                vec3 dr = {pj.pos - pi.pos};

                // Distance squared with softening
                double r2 = dr.magnitudeSquared() + softening * softening;
                double r = sqrt(r2);
                double r3 = r2 * r;

                // Force magnitude
                double force_mag = G * pi.mass * pj.mass / r3;

                // Force vector
                vec3 force = dr * force_mag;

                // Newton's third law
                pi.force += force;
                pj.force += force * (-1.0);
            }
        }
    }


    void inline kick_half_step() {
        for (int i = 0; i < particlesCount; ++i) {
            Particle& p = particles[i];
            p.acc = p.force * (1.0 / p.mass);
            p.vel += p.acc * (0.5 * dt);
        }
    }

    /// Update the positions using the Leapfrog integrator
    void updatePositionsVelocities() {
        kick_half_step();
        for (int i = 0; i < particlesCount; ++i) {
            Particle& p = particles[i];
            p.pos += p.vel * dt;
        }
        calculateForces();
        kick_half_step();
    }

    void step() {
        calculateForces();
        updatePositionsVelocities();
    }

    double getTotalEnergy() const {
        double kinetic = 0.0;
        double potential = 0.0;

        for (int i = 0; i < particlesCount; ++i) {
            const Particle& p = particles[i];
            kinetic += 0.5 * p.mass * p.vel.magnitudeSquared();
        }

        for (int i = 0; i < particlesCount; ++i) {
            for (int j = i + 1; j < particlesCount; ++j) {
                const Particle& pi = particles[i];
                const Particle& pj = particles[j];

                vec3 dr = pj.pos - pi.pos;
                double r = sqrt(dr.magnitudeSquared() + softening * softening);

                potential -= G * pi.mass * pj.mass / r;
            }
        }

        return kinetic + potential;
    }

    void saveToFile(const std::string& filename, int step_number) const {
        std::ofstream file(filename, std::ios::app);
        file << "Step: " << step_number << std::endl;
        for (int i = 0; i < particlesCount; ++i) {
            const Particle& p = particles[i];
            file << i << " " << p.pos.x << " " << p.pos.y << " " << p.pos.z
                << " " << p.vel.x << " " << p.vel.y << " " << p.vel.z
                << " " << p.mass << std::endl;
        }
        file << std::endl;
    }

    void printStatus(int step, double time) const {
        double energy = getTotalEnergy();
        std::cout << "Step: " << step << ", Time: " << time
            << ", Total Energy: " << energy
            << ", Particles: " << particlesCount << std::endl;
    }

    int getParticleCount() const {
        return particlesCount;
    }
};

int main() {
    const int num_particles = 60;
    const double total_time = 20.0;
    const double dt = 0.005;
    const int output_frequency = 100;
    const int print_frequency = 500;

    ManyBodySimulation sim(dt, 1.0, 0.1);
    sim.initializeRandomParticles(num_particles, 2.0, 5.0, 0.5);

    std::cout << "Starting many-body simulation with " << sim.getParticleCount()
        << " particles (max: 100)" << std::endl;
    std::cout << "Time step: " << dt << ", Total time: " << total_time << std::endl;

    // Clear output file
    std::ofstream clear_file("simulation_output.txt");
    clear_file.close();

    double current_time = 0.0;
    int step = 0;

    while (current_time < total_time) {
        sim.step();

        if (step % output_frequency == 0) {
            sim.saveToFile("simulation_output.txt", step);
        }

        if (step % print_frequency == 0) {
            sim.printStatus(step, current_time);
        }

        current_time += dt;
        step++;
    }

    sim.saveToFile("simulation_output.txt", step);
    sim.printStatus(step, current_time);

    std::cout << "Simulation completed! Output saved to simulation_output.txt" << std::endl;

    return 0;
}
