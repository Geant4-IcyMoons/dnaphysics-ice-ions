cmake_minimum_required(VERSION 3.16...3.27)
project(dnaphysics_proton_pipeline)

option(WITH_GEANT4_VIS "Build proton pipeline with Geant4 visualization" ON)
if(WITH_GEANT4_VIS)
  find_package(Geant4 REQUIRED vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include ${Geant4_INCLUDE_DIR})

set(PROTON_SOURCES
  src/ActionInitialization.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EventAction.cc
  src/G4DNAEmfietzoglou_iceExcitationStructure.cc
  src/G4DNAEmfietzoglou_iceProtonExcitationModel.cc
  src/G4DNAEmfietzoglou_iceProtonIonisationModel.cc
  src/G4DNAEmfietzoglou_iceProtonIonisationStructure.cc
  src/ModelDataRegistry.cc
  src/PhysicsList_Proton.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SteppingAction.cc
  src/TrackingAction.cc
)

set(PROTON_HEADERS
  include/ActionInitialization.hh
  include/DetectorConstruction.hh
  include/DetectorMessenger.hh
  include/EventAction.hh
  include/G4DNAEmfietzoglou_iceExcitationStructure.hh
  include/G4DNAEmfietzoglou_iceProtonExcitationModel.hh
  include/G4DNAEmfietzoglou_iceProtonIonisationModel.hh
  include/G4DNAEmfietzoglou_iceProtonIonisationStructure.hh
  include/ModelDataRegistry.hh
  include/PhysicsList_Proton.hh
  include/PrimaryGeneratorAction.hh
  include/RunAction.hh
  include/SteppingAction.hh
  include/TrackingAction.hh
)

add_executable(dnaphysics_proton dnaphysics_proton.cc ${PROTON_SOURCES} ${PROTON_HEADERS})
target_link_libraries(dnaphysics_proton ${Geant4_LIBRARIES})

set(MAC_FILES
  e1_proton.mac
  vis.mac
)

foreach(_script ${MAC_FILES})
  configure_file(${_script} ${PROJECT_BINARY_DIR}/. COPYONLY)
endforeach()

install(TARGETS dnaphysics_proton DESTINATION bin)
