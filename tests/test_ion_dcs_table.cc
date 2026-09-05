// Check the production sampler against analytic piecewise-linear densities.
#include "G4DNAEmfietzoglou_iceProtonDcsTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void Near(double actual, double expected, double tolerance)
{
  if (std::abs(actual - expected) > tolerance)
    throw std::runtime_error("Mismatch: " + std::to_string(actual)
                             + " expected " + std::to_string(expected));
}
}

int main()
{
  const auto directory = std::filesystem::temp_directory_path()
      / ("ion_dcs_test_" + std::to_string(getpid()));
  std::filesystem::create_directory(directory);
  setenv("DNA_PROTON_TABLE_DIR", directory.c_str(), 1);
  const double scale = (1.e-22 / 3.343) * m * m;
  const double mass = 938.27208816 * MeV;
  G4Random::setTheSeed(12345);
  try {
    std::ofstream(directory / "total.dat") << "1e6 45\n4e6 120\n";
    std::ofstream(directory / "diff.dat")
        << "1e6 10 0\n1e6 20 1\n1e6 40 3\n"
        << "4e6 10 2\n4e6 30 2\n4e6 50 2\n4e6 70 2\n";
    G4DNAEmfietzoglou_iceProtonDcsTable table;
    table.Load("total", "diff", mass);
    Near(table.TotalCrossSection(1*MeV)/scale, 45, 1e-12);
    Near(table.TotalCrossSection(4*MeV)/scale, 120, 1e-12);
    Near(table.TotalCrossSection(2*MeV)/scale, 82.5, 1e-12);
    Near(table.TotalCrossSection(0.5*MeV), 0, 0);
    constexpr int samples = 200000;
    double mean = 0.;
    int below25 = 0;
    for (int i = 0; i < samples; ++i) {
      const double W = table.SampleTransferEnergy(1*MeV, 0)/eV;
      if (W < 25) ++below25;
      mean += W;
    }
    Near(mean/samples, 30, 0.08);
    Near(double(below25)/samples, 0.25, 0.004);
    mean = 0.;
    for (int i = 0; i < samples; ++i)
      mean += table.SampleTransferEnergy(2*MeV, 0)/eV;
    Near(mean/samples, (45.*30 + 120.*40)/165., 0.15);

    // The upper row extends past Wmax at an intermediate incident energy.
    std::ofstream(directory / "total.dat") << "1e5 190\n4e5 790\n";
    std::ofstream(directory / "diff.dat")
        << "1e5 10 1\n1e5 200 1\n4e5 10 1\n4e5 800 1\n";
    table.Load("total", "diff", mass);
    const double tau = (0.2*MeV)/mass;
    const double upper = 2.*510998.95*tau*(tau+2.);
    Near(table.TotalCrossSection(0.2*MeV)/scale, 0.5*(190+upper-10), 1e-10);
    mean = 0.;
    for (int i = 0; i < samples; ++i) {
      const double W = table.SampleTransferEnergy(0.2*MeV, 0)/eV;
      if (W > upper || W < 10) throw std::runtime_error("Cutoff violation");
      mean += W;
    }
    Near(mean/samples, (190.*105+(upper-10)*(upper+10)/2)/(190+upper-10), 0.9);

    // A monoenergetic export must not access a nonexistent neighbouring row.
    std::ofstream(directory / "total.dat") << "1e6 45\n";
    std::ofstream(directory / "diff.dat") << "1e6 10 0\n1e6 40 3\n";
    table.Load("total", "diff", mass);
    Near(table.TotalCrossSection(1*MeV)/scale, 45, 1e-12);
    for (int i = 0; i < 1000; ++i) {
      const double W = table.SampleTransferEnergy(1*MeV, 0)/eV;
      if (W < 10 || W > 40) throw std::runtime_error("Single-row sampling failed");
    }
    std::cout << "PASS: node/intermediate TCS, unequal W grids, linear-density CDF, "
                 "stopping moments, Wmax, single-row tables\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(directory);
    return 1;
  }
  std::filesystem::remove_all(directory);
}
