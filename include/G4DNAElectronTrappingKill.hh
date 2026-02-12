//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .        *
// ********************************************************************
//
/// \file G4DNAElectronTrappingKill.hh
/// \brief Kill low-energy electrons in Ice and locally deposit remaining energy

#ifndef G4DNAElectronTrappingKill_h
#define G4DNAElectronTrappingKill_h 1

#include "G4ParticleChange.hh"
#include "G4VDiscreteProcess.hh"
#include "globals.hh"

class G4ParticleDefinition;
class G4Step;
class G4Track;
class G4VParticleChange;

class G4DNAElectronTrappingKill : public G4VDiscreteProcess
{
  public:
    explicit G4DNAElectronTrappingKill(
      const G4String& processName = "e-_G4DNAElectronTrappingKill_ICE");
    ~G4DNAElectronTrappingKill() override = default;

    G4bool IsApplicable(const G4ParticleDefinition& particle) override;

    G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step&) override;

    void SetKillEnergyThreshold(G4double threshold);
    G4double GetKillEnergyThreshold() const { return fKillEnergyThreshold; }

  private:
    G4double GetMeanFreePath(const G4Track& track, G4double previousStepSize,
                             G4ForceCondition* condition) override;
    G4bool IsInsideIce(const G4Track& track) const;

    G4double fKillEnergyThreshold;
    mutable G4ParticleChange fParticleChange;
};

#endif  // G4DNAElectronTrappingKill_h
