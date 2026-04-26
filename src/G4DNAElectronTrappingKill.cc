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
/// \file G4DNAElectronTrappingKill.cc
/// \brief Implementation of the G4DNAElectronTrappingKill process

#include "G4DNAElectronTrappingKill.hh"

#include <cfloat>

#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

G4DNAElectronTrappingKill::G4DNAElectronTrappingKill(const G4String& processName)
  : G4VDiscreteProcess(processName), fKillEnergyThreshold(2.0 * eV)
{}

G4bool G4DNAElectronTrappingKill::IsApplicable(const G4ParticleDefinition& particle)
{
  return (&particle == G4Electron::Definition());
}

void G4DNAElectronTrappingKill::SetKillEnergyThreshold(G4double threshold)
{
  fKillEnergyThreshold = (threshold > 0.0) ? threshold : 0.0;
}

G4double G4DNAElectronTrappingKill::GetMeanFreePath(const G4Track& track,
                                                     G4double,
                                                     G4ForceCondition* condition)
{
  *condition = NotForced;

  if (fKillEnergyThreshold <= 0.0) return DBL_MAX;
  if (track.GetDefinition() != G4Electron::Definition()) return DBL_MAX;
  if (!IsInsideIce(track)) return DBL_MAX;

  return (track.GetKineticEnergy() <= fKillEnergyThreshold) ? 0.0 : DBL_MAX;
}

G4VParticleChange* G4DNAElectronTrappingKill::PostStepDoIt(const G4Track& track,
                                                            const G4Step&)
{
  fParticleChange.Initialize(track);

  if (fKillEnergyThreshold <= 0.0) return &fParticleChange;
  if (track.GetDefinition() != G4Electron::Definition()) return &fParticleChange;
  if (!IsInsideIce(track)) return &fParticleChange;

  const G4double kinetic = track.GetKineticEnergy();
  if (kinetic > fKillEnergyThreshold) return &fParticleChange;

  if (kinetic > 0.0) {
    fParticleChange.ProposeLocalEnergyDeposit(kinetic);
  }
  fParticleChange.ProposeEnergy(0.0);
  fParticleChange.ProposeTrackStatus(fStopAndKill);
  return &fParticleChange;
}

G4bool G4DNAElectronTrappingKill::IsInsideIce(const G4Track& track) const
{
  const auto* volume = track.GetVolume();
  return (volume && volume->GetName() == "Ice");
}

