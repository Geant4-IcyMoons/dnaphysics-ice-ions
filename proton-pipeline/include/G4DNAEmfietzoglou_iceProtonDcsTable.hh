//
// Table reader and sampler for generated proton ice DCS/TCS files.
//

#ifndef G4DNAEmfietzoglou_iceProtonDcsTable_h
#define G4DNAEmfietzoglou_iceProtonDcsTable_h 1

#include "globals.hh"

#include <string>
#include <vector>

class G4DNAEmfietzoglou_iceProtonDcsTable
{
public:
  void Load(const std::string& totalTableBase,
            const std::string& diffTableFile);

  G4double TotalCrossSection(G4double kineticEnergy,
                             G4int component = -1) const;
  G4int SelectComponent(G4double kineticEnergy) const;
  G4double SampleTransferEnergy(G4double kineticEnergy,
                                G4int component) const;

  G4int NumberOfComponents() const { return fNComponents; }
  const std::string& TotalPath() const { return fTotalPath; }
  const std::string& DiffPath() const { return fDiffPath; }

private:
  struct TotalRow
  {
    G4double energy = 0.;
    std::vector<G4double> xs;
  };

  struct DiffGrid
  {
    G4double energy = 0.;
    std::vector<G4double> transfer;
    std::vector<std::vector<G4double>> dcs;
  };

  static std::string ResolvePath(const std::string& filename);
  static G4double Interpolate(G4double x1, G4double x2, G4double x,
                              G4double y1, G4double y2);
  static G4int LowerIndex(const std::vector<G4double>& values, G4double x);

  std::vector<G4double> InterpolatedDcs(G4double kineticEnergy,
                                        G4int component) const;

  std::vector<TotalRow> fTotalRows;
  std::vector<DiffGrid> fDiffGrids;
  std::string fTotalPath;
  std::string fDiffPath;
  G4int fNComponents = 0;
};

#endif
